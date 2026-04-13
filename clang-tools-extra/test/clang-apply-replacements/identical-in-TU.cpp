// RUN: mkdir -p %t.dir/Inputs/identical-in-TU

// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/identical-in-TU/identical-in-TU.cpp > %t.dir/Inputs/identical-in-TU/identical-in-TU.cpp
// RUN: sed "s#\$(path)#%/t.dir/Inputs/identical-in-TU#" %S/Inputs/identical-in-TU/file1.yaml > %t.dir/Inputs/identical-in-TU/file1.yaml
// RUN: sed "s#\$(path)#%/t.dir/Inputs/identical-in-TU#" %S/Inputs/identical-in-TU/file2.yaml > %t.dir/Inputs/identical-in-TU/file2.yaml
// RUN: clang-apply-replacements %t.dir/Inputs/identical-in-TU
// RUN: FileCheck -input-file=%t.dir/Inputs/identical-in-TU/identical-in-TU.cpp %S/Inputs/identical-in-TU/identical-in-TU.cpp

// Similar to identical test but each yaml file contains the same fix twice. 
// This check ensures that only the duplicated replacements in a single yaml 
// file are applied twice. Addresses PR45150.
