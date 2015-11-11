// DumpForDebugger.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------

import Swift

func $__lldb__DumpForDebugger_impl<TargetStream : OutputStreamType>(
    x_opt: Any?,
    _ mirror: Mirror,
    _ name: String?,
    _ indent: Int,
    _ maxDepth: Int,
    _ root: Bool,
    _ childOfCollection: Bool,
    inout _ refsAlreadySeen : Set<ObjectIdentifier>,
    inout _ maxItemCounter: Int,
    inout _ targetStream: TargetStream) {
        
        func idForObject(x: Any) -> ObjectIdentifier? {
            if let ao = x as? AnyObject {
                return ObjectIdentifier(ao)
            } else {
                return nil
            }
        }
        
        func asNumericValue(x: Any) -> Int {
            if let ao = x as? AnyObject {
                return unsafeBitCast(ao, Int.self)
            } else {
                return 0
            }
        }
        
        func isCollectionMirror(x_mirror: Mirror) -> Bool {
            let ds = x_mirror.displayStyle ?? .Struct
            switch ds {
            case .Collection:
                fallthrough
            case .Dictionary:
                fallthrough
            case .Set:
                return true
            default:
                return false
            }
        }

        func stringForObject(x_opt: Any?,
                             _ x_mirror: Mirror,
                             _ x_mirror_count: Int) -> String? {
            let ds = x_mirror.displayStyle ?? .Struct
            switch ds {
            case .Optional:
                if x_mirror_count > 0 {
                    return "\(x_mirror.subjectType)"
                }
                else {
                    if let x = x_opt {
                        return String(reflecting: x)
                    }
                }
            case .Collection:
                fallthrough
            case .Dictionary:
                fallthrough
            case .Set:
                fallthrough
            case .Tuple:
                return "\(Int(x_mirror.children.startIndex.distanceTo(x_mirror.children.endIndex))) elements"
            case .Struct:
                fallthrough
            case .Enum:
                if let x = x_opt {
                    if let cdsc = (x as? CustomDebugStringConvertible) {
                        return cdsc.debugDescription
                    }
                    if let csc = (x as? CustomStringConvertible) {
                        return csc.description
                    }
                }
                if x_mirror_count > 0 {
                    return "\(x_mirror.subjectType)"
                }
            case .Class:
                if let x = x_opt {
                    if let cdsc = (x as? CustomDebugStringConvertible) {
                        return cdsc.debugDescription
                    }
                    if let csc = (x as? CustomStringConvertible) {
                        return csc.description
                    }
                    // for a Class with no custom summary, mimic the Foundation default
                    return "<\(x.dynamicType): 0x\(String(asNumericValue(x), radix: 16, uppercase: false))>"
                } else {
                    // but if I can't provide a value, just use the type anyway
                    return "\(x_mirror.subjectType)"
                }
            default: ()
            }
            if let x = x_opt {
                return String(reflecting: x)
            }
            return nil
        }

        func ivarCount(x: Mirror) -> Int {
            let count = Int(x.children.startIndex.distanceTo(x.children.endIndex))
            if let sc = x.superclassMirror() {
                return ivarCount(sc) + count
            } else {
                return count
            }
        }

        func shouldPrint(root: Bool,
                         _ isChildOfCollection: Bool,
                         _ x: Mirror) -> Bool {
            if root || isChildOfCollection { return true }
            let count = Int(x.children.startIndex.distanceTo(x.children.endIndex))
            let sc = x.superclassMirror()
            if count > 0 { return true }
            if sc == nil { return true }
            return ivarCount(sc!) > 0
        }
        
        
        if maxItemCounter <= 0 { return }
        if !shouldPrint(root, childOfCollection, mirror) { return }
        --maxItemCounter
        
        for _ in 0..<indent { print(" ", terminator: "", toStream: &targetStream) }
        
        // do not expand classes with no custom Mirror
        // yes, a type can lie and say it's a class when it's not since we only
        // check the displayStyle - but then the type would have a custom Mirror
        // anyway, so there's that...
        var willExpand = true
        if let ds = mirror.displayStyle {
            if ds == .Class {
                if let x = x_opt {
                    if !(x is CustomReflectable) { willExpand = false }
                }
            }
        }

        let count = Int(mirror.children.startIndex.distanceTo(mirror.children.endIndex))
        let bullet = root && (count == 0 || willExpand == false) ? ""
            : count == 0    ? "- "
            : maxDepth <= 0 ? "▹ " : "▿ "
        print("\(bullet)", terminator: "", toStream: &targetStream)
        
        let needColon: Bool

        let isCollection: Bool = isCollectionMirror(mirror)
        
        if let nam = name {
            print("\(nam) ", terminator: "", toStream: &targetStream)
            needColon = true
        } else {
            needColon = false
        }
        if let str = stringForObject(x_opt, mirror, count) {
            if needColon { print(": ", terminator: "", toStream: &targetStream) }
            print("\(str)", terminator: "", toStream: &targetStream)
        }
        
        if ((maxDepth <= 0) || (false == willExpand)) { print("", toStream: &targetStream); return }

        if let x = x_opt {
            if let x_id = idForObject(x) {
                if refsAlreadySeen.contains(x_id) {
                    print(" { ... }", toStream: &targetStream)
                    return
                } else {
                    refsAlreadySeen.insert(x_id)
                    // and keep going
                }
            }
        }

        print("", toStream: &targetStream)
        
        var i = 0
        
        if let superclass_mirror = mirror.superclassMirror() {
            $__lldb__DumpForDebugger_impl(nil,
                                          superclass_mirror,
                                          "super",
                                          indent + 2,
                                          maxDepth - 1,
                                          false,
                                          false,
                                          &refsAlreadySeen,
                                          &maxItemCounter,
                                          &targetStream)
        }
        
        for (name_opt,child) in mirror.children {
            var name: String
            if name_opt == nil { name = "\(i)" }
            else { name = name_opt! }
            if maxItemCounter <= 0 {
                for _ in 0..<(indent+4) { print(" ", terminator: "", toStream: &targetStream) }
                let remainder = count - i
                print("(\(remainder)", terminator: "", toStream: &targetStream)
                if i > 0 { print(" more", terminator: "", toStream: &targetStream) }
                if remainder == 1 {
                    print(" child)", toStream: &targetStream)
                } else {
                    print(" children)", toStream: &targetStream)
                }
                return
            }
            
            $__lldb__DumpForDebugger_impl(child,
                Mirror(reflecting: child),
                name,
                indent + 2,
                maxDepth - 1,
                false,
                isCollection,
                &refsAlreadySeen,
                &maxItemCounter,
                &targetStream)
            i++
        }
}

func $__lldb__DumpForDebugger(x: Any) -> String {
    struct Output : OutputStreamType {
        var data = ""
        mutating func _lock() {
        }
        
        mutating func _unlock() {
        }
        
        mutating func write(string: String) {
            data += string
        }
    }
    
    var maxItemCounter = Int.max
    var refs: Set<ObjectIdentifier> = Set()
    var targetStream = Output()
    $__lldb__DumpForDebugger_impl(x, Mirror(reflecting: x), nil,
        0, Int.max, true, false,
        &refs, &maxItemCounter, &targetStream)
    let output = targetStream.data
    let si = output.startIndex
    let ei = output.endIndex
    if si.distanceTo(ei) > 0 {
        // if not an empty string, it's gonna have a trailing newline by construction
        // we want to strip it, as LLDB expects 'po' output to not have a newline at the end
        let rng = Range(start: si, end: ei.advancedBy(-1))
        return output[rng]
    }
    return ""
}

