// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --doxygen --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/json/GlobalNamespace/_ZTV3Foo.json --check-prefix=CLASS
// RUN: FileCheck %s < %t/json/index.json --check-prefix=INDEX

class Foo {};

// CLASS: {"Contexts":[{"DocumentationFileName":"index","End":true,"Name":"Global Namespace","QualName":"GlobalNamespace","RelativePath":"{{.*}}","USR":"0000000000000000000000000000000000000000"}],"DocumentationFileName":"_ZTV3Foo","HasContexts":true,"InfoType":"record","IsTypedef":false,"Location":{"Filename":"{{.*}}compact.cpp","LineNumber":6},"MangledName":"_ZTV3Foo","Name":"Foo","Namespace":["GlobalNamespace"],"Path":"GlobalNamespace","TagType":"class","USR":"{{([0-9A-F]{40})}}"}

// INDEX: {"Index":[{"Name":"GlobalNamespace","QualName":"GlobalNamespace","Type":"namespace","USR":"0000000000000000000000000000000000000000"}]}
