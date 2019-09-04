protocol Generic {
    associatedtype Associated: Decodable
    var test: String { get }
}

struct DecStruct: Decodable {
    let test: String
}

struct GenericImpl: Generic {
    typealias Associated = DecStruct
    let test: String = "test test"
}

protocol Problematic {
    func problemMethod<G: Generic>(param: G, anotherParam: String)
}

extension Problematic {
    func problemMethod<G>(param: G, anotherParam: String) where G : Generic {
        print("patatino")   //%self.expect('frame var -d run-target -- param',
                            //% substrs=['(a.GenericImpl) param = (test = "test test")'])
                            //%self.expect('expr -d run-target -- param',
                            //% substrs=['(a.GenericImpl)', '= (test = "test test")'])
                            //%self.expect('frame var -d run-target -- anotherParam',
                            //% substrs=['(String) anotherParam = "just a string"'])
                            //%self.expect('expr -d run-target -- anotherParam',
                            //% substrs=['(String)', '= "just a string"'])
        getAStringAsync { string in
            print("breakpoint")
        }
    }
}

class ProblematicImpl: Problematic {}

protocol NotProblematic {
    func problemMethod<G: Generic>(param: G, anotherParam: String)
}

extension NotProblematic {
    func problemMethod<G>(param: G, anotherParam: String) where G : Generic {}
}

class NotProblematicImpl: NotProblematic {
    func problemMethod<G>(param: G, anotherParam: String) where G : Generic {
        print("patatino")   //%self.expect('frame var -d run-target -- param',
                            //% substrs=['(a.GenericImpl) param = (test = "test test")'])
                            //%self.expect('expr -d run-target -- param',
                            //% substrs=['(a.GenericImpl)', '= (test = "test test")'])
                            //%self.expect('frame var -d run-target -- anotherParam',
                            //% substrs=['(String) anotherParam = "just a string"'])
                            //%self.expect('expr -d run-target -- anotherParam',
                            //% substrs=['(String)', '= "just a string"'])
        getAStringAsync { string in
            print("breakpoint")
        }
    }
}

func getAStringAsync(completion: @escaping (String) -> ()) {
    completion("asd")
}

let useCase = ProblematicImpl()
let data = GenericImpl()
useCase.problemMethod(param: data, anotherParam: "just a string")

let useCase2 = NotProblematicImpl()
let data2 = GenericImpl()
useCase2.problemMethod(param: data2, anotherParam: "just a string")