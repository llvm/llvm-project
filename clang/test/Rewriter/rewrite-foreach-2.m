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
int INNERLOOP(void);
void END_LOOP(void);
@implementation MyList (BasicTest)
- (void)compilerTestAgainst {
  id el;
        for (el in self) 
	  { LOOP(); 
            for (id el1 in self) 
	       INNER_LOOP();

	    END_LOOP();
	  }
}
@end

