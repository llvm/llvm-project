.. _header_generation:

Generating Public and Internal headers
======================================

This is a new implementation of the previous libc header generator. The old
header generator (libc-hdrgen aka "Headergen") was based on TableGen, which
created an awkward dependency on the rest of LLVM for our build system. By
creating a new standalone Headergen we can eliminate these dependencies for
easier cross compatibility.

There are 3 main components of the new Headergen. The first component are the
YAML files that contain all the function header information and are separated by
header specification and standard. The second component are the classes that are
created for each component of the function header: macros, enumerations, types,
function, arguments, and objects. The third component is the Python script that
uses the class representation to deserialize YAML files into its specific
components and then reserializes the components into the function header. The
Python script also combines the generated header content with header definitions
and extra macro and type inclusions from the .h.def file.


Instructions
------------

Required Versions:
  - Python Version: 3.8
  - PyYAML Version: 5.1

1. Keep full-build mode on when building, otherwise headers will not be
   generated.
2. Once the build is complete, enter in the command line within the build
   directory ``ninja check-newhdrgen`` to ensure that the integration tests are
   passing.
3. Then enter in the command line ``ninja libc`` to generate headers. Headers
   will be in ``build/projects/libc/include`` or ``build/libc/include`` in a
   runtime build. Sys spec headers will be located in
   ``build/projects/libc/include/sys``.


New Headergen is turned on by default, but if you want to use old Headergen,
you can include this statement when building: ``-DLIBC_USE_NEW_HEADER_GEN=OFF``

To add a function to the YAML files, you can either manually enter it in the
YAML file corresponding to the header it belongs to or add it through the
command line.

To add through the command line:

1. Make sure you are in the llvm-project directory.

2. Enter in the command line:

   .. code-block:: none

     python3 libc/newhdrgen/yaml_to_classes.py
     libc/newhdrgen/yaml/[yaml_file.yaml] --add_function "<return_type>" <function_name> "<function_arg1, function_arg2>" <standard> <guard> <attribute>

   Example:

   .. code-block:: none

      python3 libc/newhdrgen/yaml_to_classes.py
      libc/newhdrgen/yaml/ctype.yaml --add_function "char" example_function
      "int, void, const void" stdc example_float example_attribute

   Keep in mind only the return_type and arguments have quotes around them. If
   you do not have any guards or attributes you may enter "null" for both.

3. Check the YAML file that the added function is present. You will also get a
   generated header file with the new addition in the newhdrgen directory to
   examine.

If you want to sort the functions alphabetically you can check out libc/newhdrgen/yaml_functions_sorted.py.


Testing
-------

New Headergen has an integration test that you may run once you have configured
your CMake within the build directory. In the command line, enter the following:
``ninja check-newhdrgen``. The integration test is one test that ensures the
process of YAML to classes to generate headers works properly. If there are any
new additions on formatting headers, make sure the test is updated with the
specific addition.

Integration Test can be found in: ``libc/newhdrgen/tests/test_integration.py``

File to modify if adding something to formatting:
``libc/newhdrgen/tests/expected_output/test_header.h``


Common Errors
-------------
1. Missing function specific component

   Example:

   .. code-block:: none

      "/llvm-project/libc/newhdrgen/yaml_to_classes.py", line 67, in yaml_to_classes function_data["return_type"]

   If you receive this error or any error pertaining to
   ``function_data[function_specific_component]`` while building the headers
   that means the function specific component is missing within the YAML files.
   Through the call stack, you will be able to find the header file which has
   the issue. Ensure there is no missing function specific component for that
   YAML header file.

2. CMake Error: require argument to be specified

   Example:

   .. code-block:: none

     CMake Error at:
     /llvm-project/libc/cmake/modules/LLVMLibCHeaderRules.cmake:86 (message):
     'add_gen_hdr2' rule requires GEN_HDR to be specified.
     Call Stack (most recent call first):
     /llvm-project/libc/include/CMakeLists.txt:22 (add_gen_header2)
     /llvm-project/libc/include/CMakeLists.txt:62 (add_header_macro)

   If you receive this error, there is a missing YAML file, h_def file, or
   header name within the ``libc/include/CMakeLists.txt``. The last line in the
   error call stack will point to the header where there is a specific component
   missing. Ensure the correct style and required files are present:

   | ``[header_name]``
   | ``[../libc/newhdrgen/yaml/[yaml_file.yaml]``
   | ``[header_name.h.def]``
   | ``[header_name.h]``
   | ``DEPENDS``
   |   ``{Necessary Depend Files}``

3. Command line: expected arguments

   Example:

   .. code-block:: none

     usage: yaml_to_classes.py [-h] [--output_dir OUTPUT_DIR] [--h_def_file H_DEF_FILE]
     [--add_function RETURN_TYPE NAME ARGUMENTS STANDARDS GUARD ATTRIBUTES]
     [--e ENTRY_POINTS] [--export-decls]
     yaml_file
     yaml_to_classes.py:
     error: argument --add_function: expected 6 arguments

   In the process of adding a function, you may run into an issue where the
   command line is requiring more arguments than what you currently have. Ensure
   that all components of the new function are filled. Even if you do not have a
   guard or attribute, make sure to put null in those two areas.

4. Object has no attribute

   Example:

   .. code-block:: none

     File "/llvm-project/libc/newhdrgen/header.py", line 60, in __str__ for
     function in self.functions: AttributeError: 'HeaderFile' object has no
     attribute 'functions'

   When running ``ninja libc`` in the build directory to generate headers you
   may receive the error above. Essentially this means that in
   ``libc/newhdrgen/header.py`` there is a missing attribute named functions.
   Make sure all function components are defined within this file and there are
   no missing functions to add these components.

5. Unknown type name

   Example:

   .. code-block:: none

     /llvm-project/build/projects/libc/include/sched.h:20:25: error: unknown type
     name 'size_t'; did you mean 'time_t'?
     20 | int_sched_getcpucount(size_t, const cpu_set_t*) __NOEXCEPT
      |           ^
     /llvm-project/build/projects/libc/include/llvm-libc-types/time_t.h:15:24:
     note: 'time_t' declared here
     15 | typedef __INT64_TYPE__ time_t;
     |                    ^

   During the header generation process errors like the one above may occur
   because there are missing types for a specific header file. Check the YAML
   file corresponding to the header file and make sure all the necessary types
   that are being used are input into the types as well. Delete the specific
   header file from the build folder and re-run ``ninja libc`` to ensure the
   types are being recognized.

6. Test Integration Errors

   Sometimes the integration test will fail but that
   still means the process is working unless the comparison between the output
   and expected_output is not showing. If that is the case make sure in
   ``libc/newhdrgen/tests/test_integration.py`` there are no missing arguments
   that run through the script.

   If the integration tests are failing due to mismatching of lines or small
   errors in spacing that is nothing to worry about. If this is happening while
   you are making a new change to the formatting of the headers, then
   ensure the expected output file
   ``libc/newhdrgen/tests/expected_output/test_header.h`` has the changes you
   are applying.
