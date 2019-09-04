func foo() -> String {
    return "Foo"
}


func bar() -> String {
    return foo() + "bar"
}

let b = bar()
print("testing")
