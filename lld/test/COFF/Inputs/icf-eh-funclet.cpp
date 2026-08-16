// Reduced reproducer for llvm/llvm-project#41566.
//
// Compile with MSVC to produce the C++ exception-handling funclet layout that
// lld's ICF must handle correctly:
//
//   cl.exe /c /EHsc /O2 /GS- /d2FH4- icf-eh-funclet.cpp
//
// (/d2FH4- selects the classic __CxxFrameHandler3 model, matching the original
// bug report; the same issue also affects __CxxFrameHandler4.)
//
// foo() and bar() each contain a catch block whose body is byte-for-byte
// identical -- catch Thrown and rethrow it as Wrapped -- so MSVC emits two
// identical catch funclets, each in its own associative ".text$x" comdat. A
// funclet's own unwind info (.pdata/.xdata) is emitted in comdats associative
// to the funclet's *parent* function and chains to that parent's
// exception-handling data (FuncInfo).
//
// The two parent bodies differ, so the parents (and their FuncInfo) are not
// folded. The funclet bodies are identical, so ICF would fold them -- but that
// is unsafe: the single surviving funclet would then be described by two .pdata
// entries covering the same address range but pointing at different unwind
// info, and Windows' RtlLookupFunctionEntry would pick one arbitrarily. A
// thrown exception then unwinds using the wrong parent's FuncInfo and is
// dispatched to the wrong catch handler (or terminate()).
//
// The hand-written object in ../icf-eh-funclet.s models the sections this file
// produces.

struct Thrown {
  int code;
};

struct Wrapped {
  int code;
};

// Defined elsewhere; throws Thrown when it fails.
void mayThrow(int x);

// Parent #1: a single call site before the try.
int foo(int a) {
  int acc = a;
  try {
    mayThrow(a);
    acc += 1;
  } catch (const Thrown &e) {
    throw Wrapped{e.code};
  }
  return acc;
}

// Parent #2: a different body so the two parent functions are not themselves
// folded -- only the catch funclets match.
int bar(int a, int b) {
  int acc = a * 3 + b;
  try {
    mayThrow(a);
    mayThrow(b);
    acc += 7;
  } catch (const Thrown &e) {
    throw Wrapped{e.code};
  }
  return acc;
}
