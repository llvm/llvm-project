// RUN: mkdir -p %t.dir/Inputs/order-dependent
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/order-dependent/order-dependent.cpp > %t.dir/Inputs/order-dependent/order-dependent.cpp
// RUN: sed "s#\$(path)#%/t.dir/Inputs/order-dependent#" %S/Inputs/order-dependent/file1.yaml > %t.dir/Inputs/order-dependent/file1.yaml
// RUN: sed "s#\$(path)#%/t.dir/Inputs/order-dependent#" %S/Inputs/order-dependent/file2.yaml > %t.dir/Inputs/order-dependent/file2.yaml
// RUN: sed "s#\$(path)#%/t.dir/Inputs/order-dependent#" %S/Inputs/order-dependent/expected.txt > %t.dir/Inputs/order-dependent/expected.txt
// RUN: not clang-apply-replacements %t.dir/Inputs/order-dependent > %t.dir/Inputs/order-dependent/output.txt 2>&1
// RUN: diff -b %t.dir/Inputs/order-dependent/output.txt %t.dir/Inputs/order-dependent/expected.txt
