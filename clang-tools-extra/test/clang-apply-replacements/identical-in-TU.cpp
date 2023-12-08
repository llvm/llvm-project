// RUN: mkdir -p %T/Inputs/identical-in-TU

// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/identical-in-TU/identical-in-TU.cpp > %T/Inputs/identical-in-TU/identical-in-TU.cpp
// RUN: sed "s#\$(path)#%/T/Inputs/identical-in-TU#" %S/Inputs/identical-in-TU/file1.yaml > %T/Inputs/identical-in-TU/file1.yaml
// RUN: sed "s#\$(path)#%/T/Inputs/identical-in-TU#" %S/Inputs/identical-in-TU/file2.yaml > %T/Inputs/identical-in-TU/file2.yaml
// RUN: clang-apply-replacements %T/Inputs/identical-in-TU
// RUN: FileCheck -input-file=%T/Inputs/identical-in-TU/identical-in-TU.cpp %S/Inputs/identical-in-TU/identical-in-TU.cpp

// Similar to identical test but each yaml file contains the same fix twice. 
// This check ensures that only the duplicated replacements in a single yaml 
// file are applied twice. Addresses PR45150.
