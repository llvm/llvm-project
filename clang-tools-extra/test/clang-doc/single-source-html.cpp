// RUN: clang-doc --format=html --executor=standalone %s -output=%t/docs | FileCheck %s
// CHECK: Using default asset: {{.*}}{{[\/]}}share{{[\/]}}clang