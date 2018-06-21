# RUN: %clang -cc1apinotes -yaml-to-binary -o %t.apinotesc %S/Inputs/roundtrip.apinotes
# RUN: %clang -cc1apinotes -binary-to-yaml -o %t.apinotes %t.apinotesc
# RUN: diff %S/Inputs/roundtrip.apinotes %t.apinotes

