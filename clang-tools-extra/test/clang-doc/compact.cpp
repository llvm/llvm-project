// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --doxygen --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/json/GlobalNamespace/_ZTV3Foo.json --check-prefix=CLASS
// RUN: FileCheck %s < %t/json/index.json --check-prefix=INDEX

class Foo {};

// CLASS:      {
// CLASS-SAME: "Contexts":
// CLASS-SAME: [{"DocumentationFileName":"index",
// CLASS-SAME: "End":true,
// CLASS-SAME: "Name":"Global Namespace",
// CLASS-SAME: "QualName":"GlobalNamespace",
// CLASS-SAME: "RelativePath":"{{.*}}","USR":"0000000000000000000000000000000000000000"}],
// CLASS-SAME: "DocumentationFileName":"_ZTV3Foo",
// CLASS-SAME: "HasContexts":true,
// CLASS-SAME: "InfoType":"record",
// CLASS-SAME: "IsTypedef":false,
// CLASS-SAME: "Location":{"Filename":"{{.*}}compact.cpp","LineNumber":6},
// CLASS-SAME: "MangledName":"_ZTV3Foo",
// CLASS-SAME: "Name":"Foo",
// CLASS-SAME: "Namespace":["GlobalNamespace"],
// CLASS-SAME: "Path":"GlobalNamespace",
// CLASS-SAME: "TagType":"class",
// CLASS-SAME: "USR":"{{([0-9A-F]{40})}}"}

// INDEX:      {"Index":
// INDEX-SAME: [{"Name":"GlobalNamespace",
// INDEX-SAME: "QualName":"GlobalNamespace",
// INDEX-SAME: "Type":"namespace",
// INDEX-SAME: "USR":"0000000000000000000000000000000000000000"}]}
