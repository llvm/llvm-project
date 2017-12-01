define i32 @foo() #0 {
entry:
  ret i32 0
}

@bar = weak alias i32 (), i32 ()* @foo

define i32 @call_bar() #0 {
entry:
  %call = call i32 @bar()
  ret i32 %call
}
