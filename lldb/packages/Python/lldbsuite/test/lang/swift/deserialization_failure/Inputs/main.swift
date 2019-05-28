class C {}
class D : C {}

func staticTypes(_ i: Int, _ s: String) {}
func dynamicTypes(_ c: C) {}
func genericTypes<T>(_ t: T) {}

let hello = "world"
staticTypes(23, hello)
dynamicTypes(D())
genericTypes(hello)
