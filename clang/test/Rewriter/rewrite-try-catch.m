// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -

@interface Foo @end
@interface GARF @end

void foo() {
  @try  { TRY(); } 
  @catch (...) { SPLATCH(); @throw; }
}

int main()
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

