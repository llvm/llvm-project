# RUN: %clang -cc1apinotes -yaml-to-binary -o %t.apinotesc %S/Inputs/roundtrip.apinotes
# RUN: %clang -cc1apinotes -binary-to-yaml -o %t.apinotes %t.apinotesc

# Handle the infurating '...' the YAML writer adds but the parser
# can't read.

# RUN: cp %S/Inputs/roundtrip.apinotes %t-reference.apinotes
# RUN: echo "..." >> %t-reference.apinotes
# RUN: diff %t-reference.apinotes %t.apinotes

