import Foundation

protocol MyProtocol {
    func foo() -> String
}

extension MyProtocol {
    func foo() -> String {
        return "\(self)" //%self.expect('expr -d run -- self', substrs=['NSAttributedString'])
    }
}

extension NSAttributedString: MyProtocol {}
let attributed = NSAttributedString(string: "attributed", attributes: [:]).foo()
