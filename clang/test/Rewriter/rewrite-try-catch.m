// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 -std=c99 %s -o -

@interface Foo @end
@interface GARF @end

void TRY(void);
void SPLATCH(void);
void MYTRY(void);
void MYCATCH(void);

void foo(void) {
  @try  { TRY(); } 
  @catch (...) { SPLATCH(); @throw; }
}

int main(void)
{

  @try  {
     MYTRY();
  }

  @catch (Foo* localException) {
     MYCATCH();
     @throw;
  }
  
  // no catch clause
  @try { } 
  @finally { }
}

