// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -
            
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

int LOOP(void);
@implementation MyList (BasicTest)
- (void)compilerTestAgainst {
  id el;
        for (el in self) 
	  { LOOP(); }
        for (id el1 in self) 
	  LOOP();

	for (el in (self)) 
          if (el)
            LOOP(); 

	for (el in ((self))) 
          if (el)
            LOOP(); 
}
@end

