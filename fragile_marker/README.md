## Solves issues #5 and #6.  
Steps to run the pass  
1. Shift into the `fragile_marker` folder (not required but paths will change)  

    `cd fragile_marker`  
2. Create Makefile (skip if using the Makefile provided)  

    Run the following in the directory where `CMakeLists.txt` is present  
    `cmake -DLLVM_DIR=<path to directory containing LLVM-Config.cmake>`  

    Inside `fragile_marker` this is  
    `cmake -DLLVM_DIR=../build/lib/cmake/llvm`  
3. Create the shared object file for the pass (Not required if using the `.so` file already provided. However this file will only work in Linux so this step is required for other operating systems)  

    Run
    `make`
    in the directory where `Makefile` is present.  
4. Running the pass on a C file  

    Run one of the following commands

    - To generate an IR file for the source file run  

        `<path to clang executable> -O1 -g -S -emit-llvm -fno-discard-value-names -fpass-plugin=<pass to .so file of the pass> <path to C test file> -o <name of the IR file to be generated>`  
        
        For example to run the pass on the test file when inside the `fragile_marker` directory this is (assuming plugin is saved as `FragileMarkPass.so` and the output file is named `ir_file.ll`)  

        `../build/bin/clang -O1 -g -S -emit-llvm -fno-discard-value-names -fpass-plugin=FragileMarkPass.so Tests/test.c -o ir_file.ll`  

    - To generate the object file run the same command without `-S` and `-emit-llvm` and provide the name required for the object file rather than the IR file