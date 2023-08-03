- Compile the Code in your preferred language to intermediate code (ll)
	clang -S sample.c -emit-llvm -o file.ll 
- The file “LoadStoreAnalysis.cpp” contains the code which will help us run analysis on IR.
- Now, create a file “cmakeLists.txt” which will have build configuration for our needs.
- Then cmake is run which will generate platform specific build files which can be used by the platform specific build system.
- Once generated use them and the platform specific build system(in our case make) to build the final loadable module(LoadStoreAnalysis.so)
- In the previously mentioned cpp code we would have registered a pass “loadstoreanalysispass” with the LLVM pass manager.
- Now, simply use opt and load the loadable module and also specify the pass you want to run. The opt runs the pass (if registered) and then outputs the output generated.
 	opt -load ./build/libLoadStoreAnalysisPass.so -loadstoreanalysis -S hello.ll -enable-new-pm=0
- Now we can use opt and specify the desired file name and run the analysis
-------------------------------------------------------------------------------------------------------------

PROBLEM1:
Write an LLVM analysis pass that identifies load/store instructions within each function in the program. The pass should be able to handle different memory access patterns, such as direct loads/stores, pointer arithmetic, and nested structures.

-------------------------------------------------------------------------------------------------------------

PROBLEM2:
In the pass described in #21, determine details of the memory being accessed, such as the type and size.
For each memory location being accessed, track the total amount of memory accessed and the number of times the memory is accessed (i.e. frequency).

-------------------------------------------------------------------------------------------------------------

PROBLEM3:
Provide a neat report of the memory access patterns for each function, including the memory locations being accessed, the amount of data being accessed and the number of times the location is accessed.

An example output expected from the pass would be something like:

file1.c:foo()
   myArray: 12 bytes (3 times)
   myStruct.field1: 8 bytes (2 times)
   myArray2: 4 bytes (1 time)
   myStruct.field2: 8 bytes (2 times)

file2.c:bar()
   myArray3: 24 bytes (3 times)
   myStruct.field2: 16 bytes (4 times)