// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@protocol MyProtocol @end

@interface NSObject @end

@interface NSInterm : NSObject <MyProtocol>
@end

@interface NSArray : NSInterm 
@end

@interface NSSet : NSObject <MyProtocol>
@end


@interface N1 : NSObject
@end

@interface N1() <MyProtocol>
@end

NSObject* test (int argc) {
    NSArray *array = ((void*)0);
    NSSet *set = ((void*)0);
    return (argc) ? set : array ;
}


NSObject* test1 (int argc) {
    NSArray *array = ((void*)0);
    NSSet *set = ((void*)0);
    id <MyProtocol> instance = (argc) ? array : set;
    id <MyProtocol> instance1 = (argc) ? set : array;

    N1 *n1 = ((void*)0);
    id <MyProtocol> instance2 = (argc) ? set : n1;
    id <MyProtocol> instance3 = (argc) ? n1 : array;

    NSArray<MyProtocol> *qual_array = ((void*)0);
    id <MyProtocol> instance4 = (argc) ? array : qual_array;
    id <MyProtocol> instance5 = (argc) ? qual_array : array;
    NSSet<MyProtocol> *qual_set = ((void*)0);
    id <MyProtocol> instance6 = (argc) ? qual_set : qual_array;
    id <MyProtocol> instance7 = (argc) ? qual_set : array;
    id <MyProtocol> instance8 = (argc) ? qual_array : set;
    id <MyProtocol> instance9 = (argc) ? qual_array : qual_set;


    return (argc) ? array : set;
}
