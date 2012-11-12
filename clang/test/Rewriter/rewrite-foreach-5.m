// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

void *sel_registerName(const char *);
void objc_enumerationMutation(id);

@interface MyList
- (id) allKeys;
@end
    
@implementation MyList
- (unsigned int)countByEnumeratingWithState:  (struct __objcFastEnumerationState *)state objects:  (id *)items count:(unsigned int)stackcount
{
        return 0;
}
- (id) allKeys { return 0; }
@end

@interface MyList (BasicTest)
- (void)compilerTestAgainst;
@end

int LOOP();
@implementation MyList (BasicTest)
- (void)compilerTestAgainst {
  MyList * el;
  int i;
        for (el in [el allKeys]) { 
		for (i = 0; i < 10; i++)
		  if (i == 5)
		   break;

		if (el == 0)
		 break;
		if (el != self)
		  continue;
		LOOP(); 
	  }

        for (id el1 in[el allKeys]) { 
	    LOOP(); 
	    for (el in self) {
	      if (el)
		continue;
	    }
	    if (el1)
	      break;
	  }
}
@end

