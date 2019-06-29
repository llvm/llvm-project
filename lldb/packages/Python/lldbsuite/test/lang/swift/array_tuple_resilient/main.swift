// Make sure we print array of tuples containing elements with
// resilient types print correctly.

import Foundation

var patatino : [(URL, Int64)] = [(URL(string: "https://github.com")!, 1001)]
var tinky : [(URL, URL)] = [(URL(string: "https://github.com")!,
                            URL(string: "https://github.com")!)]
print(patatino) //%self.expect('frame variable -d run -- patatino',
                //%             substrs=['[0] = (0 = "https://github.com", 1 = 1001)'])
                //%self.expect('expr -d run -- patatino',
                //%             substrs=['[0] = (0 = "https://github.com", 1 = 1001)'])

print(tinky)    //%self.expect('frame variable -d run -- tinky',
                //%             substrs=['[0] = (0 = "https://github.com", 1 = "https://github.com")'])
                //%self.expect('expr -d run -- tinky',
                //%             substrs=['[0] = (0 = "https://github.com", 1 = "https://github.com")'])
