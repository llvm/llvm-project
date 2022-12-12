// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --doxygen --public --executor=standalone -p %t %t/test.cpp -output=%t/docs
//   This produces two files, index.yaml and one for the record named by its USR
//   (which we don't know in advance). This checks the record file by searching
//   for a name with a 40-char USR name.
// RUN: find %t/docs -regex ".*/[0-9A-F]*.yaml" -exec cat {} ";" | FileCheck %s --check-prefix=CHECK
// RUN: rm -rf %t

class Record {
private:
	void function_private();

public:
	void function_public();
};

void Record::function_private() {}

void Record::function_public() {}

// CHECK: ---
// CHECK-NEXT: USR:             '{{([0-9A-F]{40})}}'
// CHECK-NEXT: Name:            'Record'
// CHECK-NEXT: Path:            'GlobalNamespace'
// CHECK-NEXT: Namespace:
// CHECK-NEXT:   - Type:             Namespace
// CHECK-NEXT:     Name:             'GlobalNamespace'
// CHECK-NEXT: DefLocation:
// CHECK-NEXT:   LineNumber:      [[@LINE-20]]
// CHECK-NEXT:   Filename:        '{{.*}}'
// CHECK-NEXT: TagType:         Class
// CHECK-NEXT: ChildFunctions:
// CHECK-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// CHECK-NEXT:     Name:            'function_public'
// CHECK-NEXT:     Namespace:
// CHECK-NEXT:       - Type:            Record
// CHECK-NEXT:         Name:            'Record'
// CHECK-NEXT:         USR:             '{{([0-9A-F]{40})}}'
// CHECK-NEXT:       - Type:            Namespace
// CHECK-NEXT:         Name:            'GlobalNamespace'
// CHECK-NEXT:     DefLocation:
// CHECK-NEXT:         LineNumber:      [[@LINE-23]]
// CHECK-NEXT:         Filename:        '{{.*}}'
// CHECK-NEXT:     Location:
// CHECK-NEXT:       - LineNumber:      [[@LINE-31]]
// CHECK-NEXT:         Filename:        '{{.*}}'
// CHECK-NEXT:     IsMethod:        true
// CHECK-NEXT:     Parent:
// CHECK-NEXT:         Type:            Record
// CHECK-NEXT:         Name:            'Record'
// CHECK-NEXT:         USR:             '{{([0-9A-F]{40})}}'
// CHECK-NEXT:     ReturnType:
// CHECK-NEXT:       Type:
// CHECK-NEXT:         Name:            'void'
// CHECK-NEXT:     Access:			Public
// CHECK-NEXT: ...
