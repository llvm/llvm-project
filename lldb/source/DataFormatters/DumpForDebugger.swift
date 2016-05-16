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

func $__lldb__DumpForDebugger_impl<StreamType: Swift.OutputStream>(
    _ x_opt: Swift.Any?,
    _ mirror: Swift.Mirror,
    _ name: Swift.String?,
    _ indent: Swift.Int,
    _ maxDepth: Swift.Int,
    _ root: Swift.Bool,
    _ childOfCollection: Swift.Bool,
    _ refsAlreadySeen : inout Swift.Set<Swift.ObjectIdentifier>,
    _ maxItemCounter: inout Swift.Int,
    _ targetStream: inout StreamType) {
        
        func idForObject(_ x: Swift.Any) -> Swift.ObjectIdentifier? {
            if let ao = x as? Swift.AnyObject {
                return Swift.ObjectIdentifier(ao)
            } else {
                return nil
            }
        }
        
        func asNumericValue(_ x: Swift.Any) -> Swift.Int {
            if let ao = x as? Swift.AnyObject {
                return Swift.unsafeBitCast(ao, to: Swift.Int.self)
            } else {
                return 0
            }
        }
        
        func isCollectionMirror(_ x_mirror: Swift.Mirror) -> Swift.Bool {
            let ds = x_mirror.displayStyle ?? .`struct`
            switch ds {
            case .optional:
                fallthrough
            case .collection:
                fallthrough
            case .dictionary:
                fallthrough
            case .set:
                return true
            default:
                return false
            }
        }

        func stringForObject(_ x_opt: Swift.Any?,
                             _ x_mirror: Swift.Mirror,
                             _ x_mirror_count: Swift.Int) -> Swift.String? {
            let ds = x_mirror.displayStyle ?? .`struct`
            switch ds {
            case .optional:
                if x_mirror_count > 0 {
                    return "\(x_mirror.subjectType)"
                }
                else {
                    if let x = x_opt {
                        return String(reflecting: x)
                    }
                }
            case .collection:
                fallthrough
            case .dictionary:
                fallthrough
            case .set:
                fallthrough
            case .tuple:
                return "\(Swift.Int(x_mirror.children.count)) elements"
            case .`struct`:
                fallthrough
            case .`enum`:
                if let x = x_opt {
                    if let cdsc = (x as? Swift.CustomDebugStringConvertible) {
                        return cdsc.debugDescription
                    }
                    if let csc = (x as? Swift.CustomStringConvertible) {
                        return csc.description
                    }
                }
                if x_mirror_count > 0 {
                    return "\(x_mirror.subjectType)"
                }
            case .`class`:
                if let x = x_opt {
                    if let cdsc = (x as? Swift.CustomDebugStringConvertible) {
                        return cdsc.debugDescription
                    }
                    if let csc = (x as? Swift.CustomStringConvertible) {
                        return csc.description
                    }
                    // for a Class with no custom summary, mimic the Foundation default
                    return "<\(x.dynamicType): 0x\(Swift.String(asNumericValue(x), radix: 16, uppercase: false))>"
                } else {
                    // but if I can't provide a value, just use the type anyway
                    return "\(x_mirror.subjectType)"
                }
            default: ()
            }
            if let x = x_opt {
                return Swift.String(reflecting: x)
            }
            return nil
        }

        func ivarCount(_ x: Swift.Mirror) -> Swift.Int {
            let count = Swift.Int(x.children.count)
            if let sc = x.superclassMirror {
                return ivarCount(sc) + count
            } else {
                return count
            }
        }

        func shouldPrint(_ root: Swift.Bool,
                         _ isChildOfCollection: Swift.Bool,
                         _ x: Swift.Mirror) -> Swift.Bool {
            if root || isChildOfCollection { return true }
            let count = Swift.Int(x.children.count)
            let sc = x.superclassMirror
            if count > 0 { return true }
            if sc == nil { return true }
            return ivarCount(sc!) > 0
        }
        
        
        if maxItemCounter <= 0 { return }
        if !shouldPrint(root, childOfCollection, mirror) { return }
        maxItemCounter -= 1
        
        for _ in 0..<indent { Swift.print(" ", terminator: "", to: &targetStream) }
        
        // do not expand classes with no custom Mirror
        // yes, a type can lie and say it's a class when it's not since we only
        // check the displayStyle - but then the type would have a custom Mirror
        // anyway, so there's that...
        var willExpand = true
        if let ds = mirror.displayStyle {
            if ds == .`class` {
                if let x = x_opt {
                    if !(x is Swift.CustomReflectable) { willExpand = false }
                }
            }
        }
    
        let count = Swift.Int(mirror.children.count)
        let bullet = root && (count == 0 || willExpand == false) ? ""
            : count == 0    ? "- "
            : maxDepth <= 0 ? "▹ " : "▿ "
        Swift.print("\(bullet)", terminator: "", to: &targetStream)
        
        let needColon: Swift.Bool

        let isCollection: Swift.Bool = isCollectionMirror(mirror)
        
        if let nam = name {
            Swift.print("\(nam) ", terminator: "", to: &targetStream)
            needColon = true
        } else {
            needColon = false
        }
        if let str = stringForObject(x_opt, mirror, count) {
            if needColon { Swift.print(": ", terminator: "", to: &targetStream) }
            Swift.print("\(str)", terminator: "", to: &targetStream)
        }
        
        if ((maxDepth <= 0) || (false == willExpand)) { Swift.print("", to: &targetStream); return }

        if let x = x_opt {
            if let x_id = idForObject(x) {
                if refsAlreadySeen.contains(x_id) {
                    Swift.print(" { ... }", to: &targetStream)
                    return
                } else {
                    refsAlreadySeen.insert(x_id)
                    // and keep going
                }
            }
        }

        Swift.print("", to: &targetStream)
        
        var i = 0
        
        if let superclass_mirror = mirror.superclassMirror {
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
            var name: Swift.String
            if name_opt == nil { name = "\(i)" }
            else { name = name_opt! }
            if maxItemCounter <= 0 {
                for _ in 0..<(indent+4) { Swift.print(" ", terminator: "", to: &targetStream) }
                let remainder = count - i
                Swift.print("(\(remainder)", terminator: "", to: &targetStream)
                if i > 0 { Swift.print(" more", terminator: "", to: &targetStream) }
                if remainder == 1 {
                    Swift.print(" child)", to: &targetStream)
                } else {
                    Swift.print(" children)", to: &targetStream)
                }
                return
            }
            
            $__lldb__DumpForDebugger_impl(child,
                Swift.Mirror(reflecting: child),
                name,
                indent + 2,
                maxDepth - 1,
                false,
                isCollection,
                &refsAlreadySeen,
                &maxItemCounter,
                &targetStream)
            i += 1
        }
}

func $__lldb__DumpForDebugger(_ x: Swift.Any) -> Swift.String {
    class Output : Swift.OutputStream {
        var data = ""
        func _lock() {
        }
        
        func _unlock() {
        }
        
        func write(_ string: Swift.String) {
            data += string
        }
    }
    
    var maxItemCounter = Swift.Int.max
    var refs: Swift.Set<Swift.ObjectIdentifier> = Set()
    var targetStream = Output()
    $__lldb__DumpForDebugger_impl(x,
                                  Swift.Mirror(reflecting: x),
                                  nil,
                                  0,
                                  maxItemCounter,
                                  true,
                                  false,
                                  &refs,
                                  &maxItemCounter,
                                  &targetStream)
    var output = targetStream.data
    if output.characters.count > 0 {
        // if not an empty string, it's gonna have a trailing newline by construction
        // we want to strip it, as LLDB expects 'po' output to not have a newline at the end
        // output.remove(at: output.endIndex)
        return output
    }
    return ""
}

