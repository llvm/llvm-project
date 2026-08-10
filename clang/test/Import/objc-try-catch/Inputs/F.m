@interface Exception
@end
@interface OtherException
@end

void f() {
  @try {
    Exception *e;
    @throw e;
  }
  @catch (Exception *varname) {
  }
  @finally {
  }

  @try {
  }
  @catch (Exception *varname1) {
    @throw;
  }
  @catch (OtherException *varname2) {
  }

  @try {
  }
  @finally {
  }
}
