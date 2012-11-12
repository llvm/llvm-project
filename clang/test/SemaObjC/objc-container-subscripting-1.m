// RUN: %clang_cc1  -fsyntax-only -verify %s

typedef unsigned int size_t;
@protocol P @end

@interface NSMutableArray
@end

@interface XNSMutableArray
@end

int main() {
id array;
id oldObject = array[10]; // expected-warning {{instance method '-objectAtIndexedSubscript:' not found (return type defaults to 'id')}}

array[10] = 0; // expected-warning {{instance method '-setObject:atIndexedSubscript:' not found (return type defaults to 'id')}}

id<P> p_array;
oldObject = p_array[10]; // expected-warning {{instance method '-objectAtIndexedSubscript:' not found (return type defaults to 'id')}}

p_array[10] = 0; // expected-warning {{instance method '-setObject:atIndexedSubscript:' not found (return type defaults to 'id')}}
}

