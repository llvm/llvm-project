// RUN: %clang_cc1 -fsyntax-only %s

typedef struct objc_class *Class;
typedef struct objc_object {
 Class isa;
} *id;
    
            
@protocol P @end

@interface MyList
@end
    
@implementation MyList
- (unsigned int)countByEnumeratingWithState:  (struct __objcFastEnumerationState *)state objects:  (id *)items count:(unsigned int)stackcount
{
        return 0;
}
@end

@interface MyList (BasicTest)
- (void)compilerTestAgainst;
@end

@implementation MyList (BasicTest)
- (void)compilerTestAgainst {
	int i;
        for (id elem in self) 
           ++i;
        for (MyList *elem in self) 
           ++i;
        for (id<P> se in self) 
           ++i;

	MyList<P> *p;
        for (p in self) 
           ++i;

	for (p in p)
	  ++i;
}
@end

