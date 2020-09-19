# Decision Forest Code Completion Model

## Decision Forest
A **decision forest** is a collection of many decision trees. A **decision tree** is a full binary tree that provides a quality prediction for an input (code completion item). Internal nodes represent a **binary decision** based on the input data, and leaf nodes represent a prediction.

In order to predict the relevance of a code completion item, we traverse each of the decision trees beginning with their roots until we reach a leaf. 

An input (code completion candidate) is characterized as a set of **features**, such as the *type of symbol* or the *number of existing references*.

At every non-leaf node, we evaluate the condition to decide whether to go left or right. The condition compares one *feature** of the input against a constant. The condition can be of two types:
- **if_greater**: Checks whether a numerical feature is **>=** a **threshold**.
- **if_member**: Check whether the **enum** feature is contained in the **set** defined in the node.

A leaf node contains the value **score**.
To compute an overall **quality** score, we traverse each tree in this way and add up the scores.

## Model Input Format
The input model is represented in json format.

### Features
The file **features.json** defines the features available to the model. 
It is a json list of features. The features can be of following two kinds.

#### Number
```
{
  "name": "a_numerical_feature",
  "kind": "NUMBER"
}
```
#### Enum
```
{
  "name": "an_enum_feature",
  "kind": "ENUM",
  "enum": "fully::qualified::enum",
  "header": "path/to/HeaderDeclaringEnum.h"
}
```
The field `enum` specifies the fully qualified name of the enum.
The maximum cardinality of the enum can be **32**.

The field `header` specifies the header containing the declaration of the enum.
This header is included by the inference runtime.


### Decision Forest
The file `forest.json` defines the  decision forest. It is a json list of **DecisionTree**.

**DecisionTree** is one of **IfGreaterNode**, **IfMemberNode**, **LeafNode**.
#### IfGreaterNode
```
{
  "operation": "if_greater",
  "feature": "a_numerical_feature",
  "threshold": A real number,
  "then": {A DecisionTree},
  "else": {A DecisionTree}
}
```
#### IfMemberNode
```
{
  "operation": "if_member",
  "feature": "an_enum_feature",
  "set": ["enum_value1", "enum_value2", ...],
  "then": {A DecisionTree},
  "else": {A DecisionTree}
}
```
#### LeafNode
```
{
  "operation": "boost",
  "score": A real number
}
```

## Code Generator for Inference
The implementation of inference runtime is split across:

### Code generator
The code generator `CompletionModelCodegen.py` takes input the `${model}` dir and generates the inference library: 
- `${output_dir}/{filename}.h`
- `${output_dir}/{filename}.cpp`

Invocation
```
python3 CompletionModelCodegen.py \
        --model path/to/model/dir \
        --output_dir path/to/output/dir \
        --filename OutputFileName \
        --cpp_class clang::clangd::YourExampleClass
```
### Build System
`CompletionModel.cmake` provides `gen_decision_forest` method . 
Client intending to use the CompletionModel for inference can use this to trigger the code generator and generate the inference library.
It can then use the generated API by including and depending on this library.

### Generated API for inference
The code generator defines the Example `class` inside relevant namespaces as specified in option `${cpp_class}`.

Members of this generated class comprises of all the features mentioned in `features.json`. 
Thus this class can represent a code completion candidate that needs to be scored.

The API also provides `float Evaluate(const MyClass&)` which can be used to score the completion candidate.


## Example
### model/features.json
```
[
  {
    "name": "ANumber",
    "type": "NUMBER"
  },
  {
    "name": "AFloat",
    "type": "NUMBER"
  },
  {
    "name": "ACategorical",
    "type": "ENUM",
    "enum": "ns1::ns2::TestEnum",
    "header": "model/CategoricalFeature.h"
  }
]
```
### model/forest.json
```
[
  {
    "operation": "if_greater",
    "feature": "ANumber",
    "threshold": 200.0,
    "then": {
      "operation": "if_greater",
      "feature": "AFloat",
      "threshold": -1,
      "then": {
        "operation": "boost",
        "score": 10.0
      },
      "else": {
        "operation": "boost",
        "score": -20.0
      }
    },
    "else": {
      "operation": "if_member",
      "feature": "ACategorical",
      "set": [
        "A",
        "C"
      ],
      "then": {
        "operation": "boost",
        "score": 3.0
      },
      "else": {
        "operation": "boost",
        "score": -4.0
      }
    }
  },
  {
    "operation": "if_member",
    "feature": "ACategorical",
    "set": [
      "A",
      "B"
    ],
    "then": {
      "operation": "boost",
      "score": 5.0
    },
    "else": {
      "operation": "boost",
      "score": -6.0
    }
  }
]
```
### DecisionForestRuntime.h
```
...
namespace ns1 {
namespace ns2 {
namespace test {
class Example {
public:
  void setANumber(float V) { ... }
  void setAFloat(float V) { ... }
  void setACategorical(unsigned V) { ... }

private:
  ...
};

float Evaluate(const Example&);
} // namespace test
} // namespace ns2
} // namespace ns1
```

### CMake Invocation
Inorder to use the inference runtime, one can use `gen_decision_forest` function 
described in `CompletionModel.cmake` which invokes `CodeCompletionCodegen.py` with the appropriate arguments.

For example, the following invocation reads the model present in `path/to/model` and creates 
`${CMAKE_CURRENT_BINARY_DIR}/myfilename.h` and `${CMAKE_CURRENT_BINARY_DIR}/myfilename.cpp` 
describing a `class` named `MyClass` in namespace `fully::qualified`.



```
gen_decision_forest(path/to/model
  myfilename
  ::fully::qualifed::MyClass)
```