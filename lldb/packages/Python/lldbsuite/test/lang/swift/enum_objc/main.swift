import Foundation

func test()
{
    let a : NSNumber = 0.0
    let b : NSNumber = 1.0
    let ascending = a.compare(b)
    let descending = b.compare(a)
    let same = a.compare(a)
    return //%self.expect('frame var -d run-target -- ascending', substrs=['orderedAscending'])
           //%self.expect('frame var -d run-target -- descending', substrs=['orderedDescending'])
           //%self.expect('frame var -d run-target -- same', substrs=['orderedSame'])
           //%self.expect('expr -d run-target -- ascending', substrs=['orderedAscending'])
           //%self.expect('expr -d run-target -- descending', substrs=['orderedDescending'])
           //%self.expect('expr -d run-target -- same', substrs=['orderedSame'])
}

_ = test()
