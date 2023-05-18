// RUN: mkdir -p %T/Inputs/order-dependent
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/order-dependent/order-dependent.cpp > %T/Inputs/order-dependent/order-dependent.cpp
// RUN: sed "s#\$(path)#%/T/Inputs/order-dependent#" %S/Inputs/order-dependent/file1.yaml > %T/Inputs/order-dependent/file1.yaml
// RUN: sed "s#\$(path)#%/T/Inputs/order-dependent#" %S/Inputs/order-dependent/file2.yaml > %T/Inputs/order-dependent/file2.yaml
// RUN: sed "s#\$(path)#%/T/Inputs/order-dependent#" %S/Inputs/order-dependent/expected.txt > %T/Inputs/order-dependent/expected.txt
// RUN: not clang-apply-replacements %T/Inputs/order-dependent > %T/Inputs/order-dependent/output.txt 2>&1
// RUN: diff -b %T/Inputs/order-dependent/output.txt %T/Inputs/order-dependent/expected.txt
