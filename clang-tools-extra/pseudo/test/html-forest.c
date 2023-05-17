// RUN: clang-pseudo -source %s -html-forest=%t.html
// RUN: FileCheck %s < %t.html
int main() {
}
// Sanity check for some obvious strings.
// CHECK-DAG: <body>
// CHECK-DAG: "compound-statement"
// CHECK-DAG: main
