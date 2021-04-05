# Adding and Implementing an Intrinsic Procedure in f18
This document first describes where and how to implement an intrinsic and to write tests
at the four different stages listed below. It then illustrates this with the example of
the TRIM intrinsic.

The first step should be to be sure to have a copy of the standard and have
read the requirements for the intrinsic to be implemented.
In Fortran 2018, these are listed in section 16.9.
Writing some end-to-end Fortran test cases and running them with existing compilers is
a good way to get familiar with the intrinsic, and will be useful to write regression tests later.

To add a completely new intrinsic to f18, up to 4 steps might be needed:

- Adding support in name resolution and semantic checking
- Adding support in front-end folding (only if intrinsic can be in constant expressions)
- Adding support in the runtime (if runtime is the chosen way to implement the intrinsic)
- Adding support in lowering

For most of the Fortran 2018 intrinsics, the support in the first two steps (front-end),
has been completed. It is very likely only lowering and runtime support is missing. 
Readers that only need to implement these last two steps can focus on the runtime and lowering
related sections of this document and skip the front-end related sections.


## Quickly checking the status of an intrinsic.
The front-end support can be easily tested by running `f18 -funparse` on a test using the
intrinsic and using an explicit INTRINSIC attribute statement to declare it. If an
error about unknown intrinsic is raised, the front-end support is missing. Otherwise,
name resolution is done.

To test constant folding in the front-end, if applicable, the intrinsic can be used in the initialization
of a PARAMETER. With the `f18 -funparse`, it should be observed that the parameter
initialization has been folded. If an error is thrown, the front-end  folding support
is missing.

To test lowering, `bbc -emit-fir` should raise no error about missing intrinsics on the test
case. Note that if the test case uses modern Fortran features, it is likely that it will
currently hit other TODOs. So it is better to keep the test cases simple at first (using F77 features
when possible).

To test the runtime support, it will be needed to try to compile and link the test program with the runtime.
There is no official end-to-end driver for now, so this test is a bit more cumbersome.
Have a look at `test/Examples/hello.f90` for a possible flow to do this end-to-end compilation.
Also, if the runtime is implemented but lowering support is not there, this test will also fail,
so it might just be easier to search the runtime headers in `lib/runtime` for a function that would
contain the intrinsic name, and see if it has an actual implementation in one of the `.cpp` files.

## Adding intrinsics in the front-end
### Name resolution and semantic checking
- See: `lib/Evaluate/intrinsics.cpp`.
- Where to submit patch: llvm repository through Phabricator.

#### Implementation
There is a table in this file, and intrinsics must be defined there with
their interface and argument constraints.

#### Testing
Be sure to add some error tests in the semantics regression tests.
See `test/Semantics/reshape.f90` for a good example.

### Front-end folding
- See: `lib/Evaluate/fold.cpp`, and `lib/Evaluate/fold-[type].cpp`.
- Where to submit patch: llvm repository through Phabricator.

#### Implementation
If the intrinsic that is being added is allowed in constant expressions, then an
implementation operating on `Fortran::evaluate::Expr` must be added to the
front-end. Otherwise, this step can be skipped. In Fortran 2018 standard, section
10.1.12 rules which intrinsic functions are allowed in constant expressions.

In each of those files, there is a big `if` statement switching on the
intrinsic names. Chose the file according to the intrinsic result types. If the
intrinsic may return several types and would have a similar implementation, the
implementation should be done in `lib/Evaluate/fold.cpp`.
Add the name of the intrinsics in the if-chain, and plug the implementation there.
See other implementations for examples.

Front-end folding operates on `Fortran::evaluate::Expr`, and in particular, with the
`Fortran::evaluate::Constant` variant of expressions. You will need to be
familiar with them to do anything. Look at the defining header in `include/Evaluate`
to get started. Note that `Fortran::evaluate::Expr` has `Dump` and `AsFortran`
methods that can be used to see what it contains.

#### Tests
To add regression tests for front-end folding, `test_folding.sh` can be used as a test driver.
It checks that every parameter prefixed by `test_` is folded to `.true.`.
`test/Evaluate/folding06.f90` is a good example for it.

## Adding intrinsics to the runtime and lowering

It is first needed to decide whether some runtime is needed or not. If the intrinsic can
be  implemented with a simple sequence of FIR operations, it might be worth to implement it inline.
Otherwise, if it involves more advanced control flow and dynamic allocations, a runtime implementation
might be sounder.

### Runtime
- See: `lib/runtime` folder
- Where to submit patch: llvm repository through Phabricator.

#### Implementation
A runtime interface must be designed for the intrinsic.
The runtime interface is simply the declaration of the runtime function for which lowering must insert calls,
as well as a small description of what the function is doing (e.g. comments above the declaration).
Before starting to implement anything, it is best to submit this interface for review. Having the interface
already submitted can also allow the implementation of the lowering part in parallel of the runtime implementation.
See `lib/runtime/character.h` for examples of runtime interfaces.

The design must decide if one or more functions should form the runtime interface.
It may make sense to have simple entry points for easy cases, and a more complex entry points
with descriptors to cover all the other cases. When in doubt, it is best to start with a single
entry point that cover all cases, and leave entry point specialization for later, based on a measurable
performance gain on actual programs.

The implementation is done in C++, but there is a hard requirement in not depending on
the C++ runtime library. That is because it is undesirable that Fortran runtime depends
upon libstdc++ or other C++ runtime libraries. This means that templates and classes
are OK, `#include <c[library]>` should be OK, but any other includes should be avoided
unless proven they do not bring C++ runtime in.
Trying to link the runtime (libFortranRuntime.a) and a test program with a C compiler will
leads to linking failure if there is a dependency on libsdtc++. This can be used to test
that the Fortran runtime remains independent of the C++ runtime.

The implementation is done in a related `.cpp` file in the same folder. See
`lib/runtime/character.cpp` for an example.

#### Tests
To be defined. Ideally, we would want end-to-end tests from Fortran, but it is
not yet clear where to put them in llvm. Unit tests can also be defined. See
`unittests/Runtime/character.cpp` for an example.

### Lowering
- See: `lib/Lower/IntrinsicCall.cpp`
- Where to submit patch: fir-dev branch of https://github.com/flang-compiler/f18-llvm-project.

#### Implementation
Lowering of intrinsics is driven by the `IntrinsicLibrary` class. Its member function
`gen{intrinsic name}` lowers "intrinsic name". These functions have standardized interfaces.
There is a table in this file that maps an intrinsic name to the function that lowers it.
To add support for an intrinsic, a new member function containing the lowering implementation
must be created and mapped into the table.

The lowering implementation can be inlined if it can easily be implemented in a few fir or
builtin mlir dialect operations. Otherwise, the lowering implementation should generate code
that emits a call to the runtime implementing the intrinsic.

There are three standardized interfaces to choose from for implementation member functions:
- `mlir::Value genXXX(mlir::Type, llvm::ArrayRef<mlir::Value>)`
- `fir::ExtendedValue genXXX(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>)`
- `void genXXX(llvm::ArrayRef<fir::ExtendedValue>)`

The first interface is to be used for elemental numerical and logical intrinsic functions,
see `genAbs` for an example.
The second interface is to be used for other intrinsic functions, see `genTrim` for an example.
The third interface is to be used for intrinsic subroutines, see `genDateAndTime` for an example.

It is worth being familiar with `fir::ExtendedValue`. It describes an entity (symbol,
or evaluated expression) with all its associated dynamic properties (length, extents, lower bounds...).
A `fir::ExtendedValue` is a variant of different SSA value containers that describes a particular category of entities.
The categories are:
- `fir::Unboxed` for scalar numericals, logicals, and non-polymorphic derived types without length parameters,
- `fir::CharBoxValue` for scalar characters.
- `fir::ArrayBoxValue` for contiguous arrays of numerical, logical, and non-polymorphic derived types without length parameters.
- `fir::CharArrayBoxValue` contiguous character arrays.
- `fir::MutableBoxValue` for allocatable and pointers of all types.
- `fir::ProcBoxValue` for procedure pointers.
- `fir::BoxValue` for all the rest (e.g. non-contiguous arrays, polymorphic or parametrized derived types).
See `include/flang/Support/BoxValue.h` for more details.

If the intrinsic has a runtime implementation, the actual binding should be done in another helper function called
`gen{Runtime function name}`. For instance character runtime bindings are implemented in `lib/Lower/CharacterRuntime.cpp`
See `genTrim` for an example of such runtime binding helper.
Non character related runtime binding should be added to `lib/Lower/IntrinsicRuntime.cpp`.

Note: Numerical elemental intrinsics that have libm/libnvmath implementations are automatically mapped and
do not follow the above patterns (this includes intrinsics like `acos`, `bessel_j`...).

#### Tests
LLVM FileCheck regression tests are used in lowering. See `test/Lower/intrinsic-procedures.f90`.
To get familiar with FileCheck tests, it is worth having a look at `https://llvm.org/docs/CommandGuide/FileCheck.html`.
A few test cases must be added in `intrinsic-procedures.f90` to test that the fir output looks like what is expected.
To do quick eye checking during development, the FIR produced for a Fortran file can be dumped with `bbc -emit-fir`.
The `-emit-fir` option ensures `bbc` does not run any simplification passes after lowering. These passes could make
it harder to check what FIR was actually generated by lowering.

## Tips

Be sure to respect alphabetical order when adding a new intrinsic in one of the tables or
if statements. They are usually sorted according to intrinsic names.

## An End-to-End example: TRIM
TRIM is a transformational intrinsic operating on characters.
It takes a scalar character and returns a scalar character. See Fortran standard for more details about it.
### TRIM name resolution and semantic checking
In lib/Evaluate/intrinsics.cpp, the table contains a line:
```c++
 {"trim", {{"string", SameChar, Rank::scalar}}, SameChar, Rank::scalar, IntrinsicClass::transformationalFunction}
```
This defines a transformational intrinsic named "trim" that takes a scalar character argument
named "string" and returns a scalar character of the same kind as the argument.
There is nothing more needed for name resolution and semantic checking.

### TRIM front-end folding
TRIM can appear in constant expressions, as per Fortran 2018 section 10.1.12 point 1 sub-point (6).

In `lib/Evaluate/fold-character.cpp`, there is case to fold trim that looks like:
```c++
  } else if (name == "trim") { // not elemental
    if (auto scalar{
            GetScalarConstantArguments<T>(context, funcRef.arguments())}) {
      return Expr<T>{Constant<T>{
          CharacterUtils<KIND>::TRIM(std::get<Scalar<T>>(*scalar))}};
    }
  }
```
It uses a helper function operating on the `Scalar` abstract representation. When T is a character kind 1, `Scalar<T>`
are simply `std::string`, so `CharacterUtils<1>::TRIM` actually simply implements TRIM on `std::string`.
The folding code above mainly tries to extract a constant from the argument, and if successful, repackages the result
of the helper in a new typed expression representation (`Expr<T>`).

Regression tests for TRIM can be found in `test/Evaluate/folding05.f90` and look like:

```fortran
logical, parameter :: test_c1_trim2 = trim('ab  ') .eq. 'ab'
```

A test script is running the front-end on the test file and checking that `test_c1_trim2` has been
folded to `.true.` in the unparsed output. To add a new test case, add a new logical parameter
that must fold to `.true.` with a name prefixed by `test_`.

### TRIM runtime
TRIM runtime API is defined in `runtime/character.h` as:

```c++
void RTNAME(Trim)(Descriptor &result, const Descriptor &string,
    const char *sourceFile = nullptr, int sourceLine = 0);
```

The `RTNAME` macro adds some mangling to the runtime to avoid collision with user symbols and handles
runtime library versioning. Here, it only matters to know that it must be used in all function
declarations the runtime API.

TRIM runtime takes a descriptor for the result that it will allocate, and a constant descriptor for the string argument.
On top of that, it takes optional source file and line arguments, that are here to print source location if the runtime
was to reach a critical error (e.g., if the provided descriptor is actually not describing a character scalar).
Note that C++ runtime optional arguments must still be created explicitly in lowering. They are only indications
of what the runtime expects as default values. Here, if no source location were available, lowering code would have to
generate `nullptr` and `0` and pass them explicitly in FIR.

It is rather simple and follows the standard definition of TRIM in the argument naming. In general, it is best to stick
to the intrinsic interface in terms of argument names and order if possible.

The actual implementation is in `runtime/character.cpp` and is not copied here. Have a look at it. A notable point is that it
is not using `std::string` anywhere, because that would bring the C++ runtime as a dependency.

It is using some of the `Descriptor` member functions such as `string.OffsetElement<>`,
`result.Establish()`, `result.Allocate()`, or `string.ElementBytes()`. It is worth having a look at
`flang/runtime/descriptor.h` to understand more about the descriptor format and the tools available for it.

The error handling in the runtime is done with the `Terminator` class. See the `terminator.Crash()`
and `RUNTIME_CHECK` macro. `terminator.Crash` causes an unconditional crash, while `RUNTIME_CHECK`
will cause a crash if the boolean provided to it is not true. The runtime should always crash rather
than reach undefined states.

The last notable point is that it is using LEN_TRIM implementation rather than re-implementing it locally.
It is always a good idea to share parts of implementations that are similar between related intrinsics.
It is OK to define internal helper functions, classes or to use templates to achieve this.


### TRIM Lowering

In the table in `lib/Lower/IntrinsicCall.cpp`, there is a line:
```c++
{"trim", &I::genTrim, {{{"string", asAddr}}}, /*isElemental=*/false}
```

It defines that when presented with a call to an intrinsic named "trim",
that is not elemental, lowering must lower its `string` argument in memory, and pass this lowered
argument to a function called `genTrim` that implements the actual lowering of TRIM.

Pass by value is the default argument passing scheme. If all arguments are passed by value, then you do not need to specify any arguments in this table. If at least one argument is not passed by value, then it is good practice to specify all the arguments. For example, consider the genScan entry:
```c++
{"scan", &I::genScan, {{ {"string", asAddr}, {"set", asAddr}, 
                         {"back", asValue}, {"kind", asValue} }}, 
                         /*isElemental=*/true}
```

The `genTrim` function looks like:
```c++
// TRIM
fir::ExtendedValue
IntrinsicLibrary::genTrim(mlir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
   // Have a look at the implementation in lib/Lower/IntrinsicCall.cpp directly
}
```

It provides the result type and the lowered arguments as inputs, and expects the evaluated
result to be returned. Given the argument and result types are scalar characters, the related
`fir::Extended` must be `fir::CharBoxValue`, which is simply a container over two SSA values: one
for the address, one for the length.

The choice here was to implement TRIM with the help of runtime. So this code is only preparing the arguments
according to the runtime interface. TRIM runtime is described in the next section.
It takes a `Descriptor&` for the result, and a `const Descriptor&` for the argument. The result can be seen
as a temporary allocatable that the runtime will allocate.
The code here is creating a `fir::MutableBoxValue`, which is a class used to deal with allocatables in lowering.
It also creates a descriptor for the argument with `builder.createBox()`. Note that the two descriptors are created
differently because we expect the first one to be modified, but not the second one. This difference preserves SSA
semantics in the IR (An SSA value cannot change).

The actual runtime call is generated by `Fortran::lower::genTrim` defined in `lib/Lower/CharacterRuntime.cpp`
After the runtime call, the result descriptor is read to build the resulting `fir::CharBoxValue`
with the result address and length. Since the result address was allocated, it is added to a clean-up list so that
it can be deallocated after the statement.

The implementation of the runtime call in lowering looks like:

```c++
void Fortran::lower::genTrim(Fortran::lower::FirOpBuilder &builder,
                             mlir::Location loc, mlir::Value resultBox,
                             mlir::Value stringBox) {
  auto trimFunc = getRuntimeFunc<mkRTKey(Trim)>(loc, builder);
  auto fTy = trimFunc.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(3));

  llvm::SmallVector<mlir::Value> args;
  args.emplace_back(builder.createConvert(loc, fTy.getInput(0), resultBox));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(1), stringBox));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(2), sourceFile));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(3), sourceLine));
  builder.create<fir::CallOp>(loc, trimFunc, args);
}
```

The key point is `getRuntimeFunc<mkRTKey(Trim)>(loc, builder)` that builds the FIR signature for the runtime
function automatically. The name passed to `mkRTKey` must be the same as the one inside `RTNAME` when declaring
the function in the runtime headers. The runtime header must be included in the current file to use `getRuntimeFunc<>`
(note the `#include "../../runtime/character.h"` at the top of the file). So at least the runtime API must be designed before adding
the support in lowering.

Then, the source file name and line number are lowered from the current location so that they can be passed to the runtime.
In general, runtime calls that may fail (like here if the allocation were to fail) takes source file information.

Last, before making the call, all SSA values that were lowered from the arguments are converted to the exact type required by
the runtime. There are two reasons for this. The first is that the runtime uses a single opaque type for descriptors,
but fir descriptor are strongly typed. So all descriptors must be cast to this opaque type (this will be a no-op at runtime).
The second reason is that when intrinsics take integer arguments, the actual integer type can be of any kind, but the runtime usually will take
a simple `std::int64_t` argument to cover all cases (assuming in these cases that bigger integer values would not be
semantically valid.). So an actual truncation/extension might be required. Therefore, casts are systematically inserted for
runtime arguments to simplify interfaces.

That is it. Here is the fir output of TRIM lowering for a simple Fortran program:
```fortran
   character(42) :: c
   call bar(trim(c))
  end
```

```
func @_QQmain() {
  // Allocation of the descriptor for the temporary result of TRIM.
  // This was generated by the `createTempMutableBox` call
  %0 = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
  
  // Getting the address of c. This was actually generated when lowering the
  // program scope, not by TRIM intrinsic lowering. It is the address in the fir::CharBoxValue argument.
  %1 = fir.address_of(@_QEc) : !fir.ref<!fir.char<1,42>>
  
  // Creating a descriptor for c. This was generated by the `createBox()` call.
  // Note that the descriptor allocation is not visible in FIR because this is a read-only
  // descriptor that cannot be modified after its creation.
  %2 = fir.embox %1 : (!fir.ref<!fir.char<1,42>>) -> !fir.box<!fir.char<1,42>>
  
  // These ops are initializing the result descriptor to an unallocted descriptor
  // of character type. They were also generated by `createTempMutableBox`.
  %3 = fir.zero_bits !fir.heap<!fir.char<1,?>>
  %c0 = constant 0 : index
  %4 = fir.embox %3 typeparams %c0 : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
  fir.store %4 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  
  // This is the address of the global constant with the file name. It was generated by `locationToFilename`.
  %5 = fir.address_of(@_QQcl.2E2F7472696D2E66393000) : !fir.ref<!fir.char<1,11>>
  // This is the line number. It was generated by `locationToLineNo`
  %c2_i32 = constant 2 : i32
  
  // These are the argument casts generated in `Fortran::lower::genTrim`.
  %6 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  %7 = fir.convert %2 : (!fir.box<!fir.char<1,42>>) -> !fir.box<none>
  %8 = fir.convert %5 : (!fir.ref<!fir.char<1,11>>) -> !fir.ref<i8>
  
  // This is the runtime call generated in `Fortran::lower::genTrim`.
  %9 = fir.call @_FortranATrim(%6, %7, %8, %c2_i32) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  
  // This is reading the result address and length from the result descriptor, generated by `genMutableBoxRead`.
  %10 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  %11 = fir.box_elesize %10 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
  %12 = fir.box_addr %10 : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
  
  // This is the call to bar. fir.boxchar represent the F77 way of passing characters. This
  // is generated in lowering based on the fir::ExtendedValue that was returned by `genTrim`.
  %13 = fir.convert %12 : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
  %14 = fir.emboxchar %13, %11 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  fir.call @_QPbar(%14) : (!fir.boxchar<1>) -> ()
  
  // After bar, the address of the temporary TRIM result is freed. This was indirectly generated by 
  // `addCleanUpForTemp` in `genTrim`. It registered code to be generated at the end of the statement.
  fir.freemem %12 : !fir.heap<!fir.char<1,?>>
  return
}
```


Regression tests for TRIM lowering are in `test/Lower/intrinsic-procedures.f90` and look like:

```fortran
! TRIM
! CHECK-LABEL: trim_test
! CHECK-SAME: (%[[arg0:.*]]: !fir.boxchar<1>)
subroutine trim_test(c)
  character(*) :: c
  ! CHECK: %[[tmpBox:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
  ! CHECK-DAG: %[[c:.*]]:2 = fir.unboxchar %[[arg0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK-DAG: %[[cBox:.*]] = fir.embox %[[c]]#0 typeparams %[[c]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK-DAG: %[[cBoxNone:.*]] = fir.convert %[[cBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK-DAG: %[[resBox:.*]] = fir.convert %[[tmpBox]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}Trim(%[[resBox]], %[[cBoxNone]], {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK-DAG: %[[tmpAddr:.*]] = fir.box_addr
  ! CHECK-DAG: fir.box_elesize
  ! CHECK: fir.call @{{.*}}bar_trim_test
  call bar_trim_test(trim(c))
  ! CHECK: fir.freemem %[[tmpAddr]] : !fir.heap<!fir.char<1,?>>
  return
end subroutine
```

The `CHECK-XXX` are checking for patterns in the FIR dumped after lowering of a Fortran function using TRIM. The `[[xxx:.*]]`
are capturing SSA value definitions and the `[[xxx]]` are checking for their uses. See FileCheck documentation for more information.
To write such regression tests, start from a FIR dump that was manually verified to match with what is expected to be produced by lowering.
Keep the most relevant operations for the intrinsic, and replace the SSA value names with FileCheck pattern matching.
Note the usage of `CHECK-DAG` in places where the exact order of what SSA value is produced first does not matter.
Here, we do not want to make it a requirement that `resBox` value must be produced after `cBoxNone` SSA value.
It does not really matter in which order the intrinsic runtime arguments are lowered, someone should be able to
change that without breaking the test.
