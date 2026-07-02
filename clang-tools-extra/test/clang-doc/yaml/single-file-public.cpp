// RUN: rm -rf %t && mkdir -p %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%S/../Inputs/single-file-public.cpp" "%t/test.cpp"
// RUN: clang-doc --doxygen --public --executor=standalone -p %t %t/test.cpp -output=%t/docs
//   This produces two files, index.yaml and one for the record named by its USR
