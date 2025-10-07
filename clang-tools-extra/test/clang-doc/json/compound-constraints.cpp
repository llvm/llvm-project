// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --extra-arg -std=c++20 --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/json/index.json

template<typename T> concept Incrementable = requires (T a) {
  a++;
};

template<typename T> concept Decrementable = requires (T a) {
  a--;
};

template<typename T> concept PreIncrementable = requires (T a) {
  ++a;
};

template<typename T> concept PreDecrementable = requires (T a) {
  --a;
};

template<typename T> requires Incrementable<T> && Decrementable<T> void One();

template<typename T> requires (Incrementable<T> && Decrementable<T>) void Two();

template<typename T> requires (Incrementable<T> && Decrementable<T>) || (PreIncrementable<T> && PreDecrementable<T>) void Three();

template<typename T> requires (Incrementable<T> && Decrementable<T>) || PreIncrementable<T> void Four();

// CHECK:         "Name": "One",
// CHECK:         "Template": {
// CHECK-NEXT:      "Constraints": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Incrementable<T>",
// CHECK-NEXT:          "Name": "Incrementable",
// CHECK-NEXT:          "Path": "",
// CHECK-NEXT:          "QualName": "Incrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Expression": "Decrementable<T>",
// CHECK-NEXT:          "Name": "Decrementable",
// CHECK-NEXT:          "Path": "",
// CHECK-NEXT:          "QualName": "Decrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK:         "Name": "Two",
// CHECK:         "Template": {
// CHECK-NEXT:      "Constraints": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Incrementable<T>",
// CHECK-NEXT:          "Name": "Incrementable",
// CHECK-NEXT:          "Path": "",
// CHECK-NEXT:          "QualName": "Incrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Expression": "Decrementable<T>",
// CHECK-NEXT:          "Name": "Decrementable",
// CHECK-NEXT:          "Path": "",
// CHECK-NEXT:          "QualName": "Decrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK:         "Name": "Three",
// CHECK:         "Template": {
// CHECK-NEXT:      "Constraints": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Incrementable<T>",
// CHECK-NEXT:          "Name": "Incrementable",
// CHECK-NEXT:          "Path": "",
// CHECK-NEXT:          "QualName": "Incrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Decrementable<T>",
// CHECK-NEXT:          "Name": "Decrementable",
// CHECK-NEXT:          "Path": "",
// CHECK-NEXT:          "QualName": "Decrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "PreIncrementable<T>",
// CHECK-NEXT:          "Name": "PreIncrementable",
// CHECK-NEXT:          "Path": "",
// CHECK-NEXT:          "QualName": "PreIncrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Expression": "PreDecrementable<T>",
// CHECK-NEXT:          "Name": "PreDecrementable",
// CHECK-NEXT:          "Path": "",
// CHECK-NEXT:          "QualName": "PreDecrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK:         "Name": "Four",
// CHECK:         "Template": {
// CHECK-NEXT:      "Constraints": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Incrementable<T>",
// CHECK-NEXT:          "Name": "Incrementable",
// CHECK-NEXT:          "Path": "",
// CHECK-NEXT:          "QualName": "Incrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Decrementable<T>",
// CHECK-NEXT:          "Name": "Decrementable",
// CHECK-NEXT:          "Path": "",
// CHECK-NEXT:          "QualName": "Decrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Expression": "PreIncrementable<T>",
// CHECK-NEXT:          "Name": "PreIncrementable",
// CHECK-NEXT:          "Path": "",
// CHECK-NEXT:          "QualName": "PreIncrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
