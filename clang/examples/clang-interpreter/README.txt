This is an example of Clang based interpreter, for executing standalone C/C++
programs.

It demonstrates the following features:
 1. Parsing standard compiler command line arguments using the Driver library.

 2. Constructing a Clang compiler instance, using the appropriate arguments
    derived in step #1.

 3. Invoking the Clang compiler to lex, parse, syntax check, and then generate
    LLVM code.

 4. Use the LLVM JIT functionality to execute the final module.

 5. Intercepting a Win64 library call to allow throwing and catching exceptions
    in and from the JIT.

The implementation has many limitations and is not designed to be a full fledged
interpreter. It is designed to demonstrate a simple but functional use of the
Clang compiler libraries.
