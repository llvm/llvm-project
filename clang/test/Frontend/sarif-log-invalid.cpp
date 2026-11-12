// RUN: rm -rf %t && mkdir %t && %clang -c -fdiagnostics-add-output=sarif -Xclang -verify=a %s
// RUN: %clang -c -fdiagnostics-add-output=sarif:file=b.sarif,version=2.0 -Xclang -verify=b %s
// RUN: %clang -c -fdiagnostics-add-output=sarif:unknown=foo,file=c.sarif -Xclang -verify=c %s
// RUN: %clang -c -fdiagnostics-add-output=invalid -Xclang -verify=d %s

// a-error@*{{'-fdiagnostics-add-output' with format 'sarif' is missing required key 'file'}}
// b-error@*{{'-fdiagnostics-add-output' specifies unsupported SARIF version '2.0'. Supported versions are}}
// c-error@*{{'-fdiagnostics-add-output' with format 'sarif' uses unrecognized key 'foo'}}
// d-error@*{{'-fdiagnostics-add-output' uses unrecognized format 'invalid'}}
