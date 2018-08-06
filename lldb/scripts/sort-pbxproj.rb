#! /usr/bin/ruby
#

# A script to impose order on the Xcode project file, to make merging
# across branches were many additional files are present, easier.




## Sort the BuildFile and FileReference sections of an Xcode project file,
## putting Apple/github-local files at the front to avoid merge conflicts.
#
## Run this in a directory with a project.pbxproj file.  The sorted version
## is printed on standard output.
#


# Files with these words in the names will be sorted into a separate section;
# they are only present in some repositories and so having them intermixed 
# can lead to merge failures.
segregated_filenames = ["Swift", "repl", "RPC"]

def read_pbxproj(fn)
    beginning  = Array.new   # All lines before "PBXBuildFile section"
    files      = Array.new   # PBXBuildFile section lines -- sort these
    middle     = Array.new   # All lines between PBXBuildFile and PBXFileReference sections
    refs       = Array.new   # PBXFileReference section lines -- sort these
    ending     = Array.new   # All lines after PBXFileReference section

    all_lines = File.readlines fn

    state = 1 # "begin"
    all_lines.each do |l|
        l.chomp
        if state == 1 && l =~ /Begin PBXBuildFile section/
            beginning.push(l)
            state = 2
            next
        end
        if state == 2 && l =~ /End PBXBuildFile section/
            middle.push(l)
            state = 3
            next
        end
        if state == 3 && l =~ /Begin PBXFileReference section/
            middle.push(l)
            state = 4
            next
        end
        if state == 4 && l =~ /End PBXFileReference section/
            ending.push(l)
            state = 5
            next
        end

        if state == 1
            beginning.push(l)
        elsif state == 2
            files.push(l)
        elsif state == 3
            middle.push(l)
        elsif state == 4
            refs.push(l)
        else
            ending.push(l)
        end
    end

    return beginning, files, middle, refs, ending
end

xcodeproj_filename = nil
[ "../lldb.xcodeproj/project.pbxproj", "lldb.xcodeproj/project.pbxproj", "project.pbxproj" ].each do |ent|
    if File.exists?(ent)
        xcodeproj_filename = ent
        break
    end
end

if xcodeproj_filename.nil?
    STDERR.puts "Could not find xcode project file to sort."
    exit(1)
end

beginning, files, middle, refs, ending = read_pbxproj(xcodeproj_filename)


### If we're given a "canonical" project.pbxproj file, get the uuid and fileref ids for
### every source file in this project.pbxproj and the canonical one, and fix any of
### the identifiers that don't match in the project file we're updating.
### this comes up when people add the file independently on different branches and it
### gets different identifiers.

if ARGV.size() > 0
    canonical_pbxproj = nil
    if ARGV.size == 2 && ARGV[0] == "--canonical"
        canonical_pbxproj = ARGV[1]
    elsif ARGV.size == 1 && ARGV[0] =~ /--canonical=(.+)/
        canonical_pbxproj = $1
    end

    if File.exists?(canonical_pbxproj)
        ignore1, canon_files, ignore2, ignore3, ignore4 = read_pbxproj(canonical_pbxproj)
        canon_files_by_filename = Hash.new { |k, v| k[v] = Array.new }

        canon_files.each do |l|
            # 2669421A1A6DC2AC0063BE93 /* MICmdCmdTarget.cpp in Sources */ = {isa = PBXBuildFile; fileRef = 266941941A6DC2AC0063BE93 /* MICmdCmdTarget.cpp */; };

            if l =~ /^\s+([A-F0-9]{24})\s+\/\*\s+(.*?)\sin.*?\*\/.*?fileRef = ([A-F0-9]{24})\s.*$/
                uuid = $1
                filename = $2
                fileref = $3
                canon_files_by_filename[filename].push({ :uuid => uuid, :fileref => fileref })
            end
        end

        this_project_files = Hash.new { |k, v| k[v] = Array.new }

        files.each do |l|
            # 2669421A1A6DC2AC0063BE93 /* MICmdCmdTarget.cpp in Sources */ = {isa = PBXBuildFile; fileRef = 266941941A6DC2AC0063BE93 /* MICmdCmdTarget.cpp */; };

            if l =~ /^\s+([A-F0-9]{24})\s+\/\*\s+(.*?)\sin.*?\*\/.*?fileRef = ([A-F0-9]{24})\s.*$/
                uuid = $1
                filename = $2
                fileref = $3
                this_project_files[filename].push({ :uuid => uuid, :fileref => fileref })
            end
        end

        this_project_files.keys.each do |fn|
            next if !canon_files_by_filename.has_key?(fn)
            next if this_project_files[fn].size() > 1 || canon_files_by_filename[fn].size() > 1
            this_ent = this_project_files[fn][0]
            canon_ent = canon_files_by_filename[fn][0]
            if this_ent[:uuid] != canon_ent[:uuid]
                STDERR.puts "#{fn} has uuid #{this_ent[:uuid]} in this project file, #{canon_ent[:uuid]} in the canonical"
                [ beginning, files, middle, refs, ending ].each do |arr|
                    arr.each { |l| l.gsub!(this_ent[:uuid], canon_ent[:uuid]) }
                end
            end
            if this_ent[:fileref] != canon_ent[:fileref]
                STDERR.puts "#{fn} has fileref #{this_ent[:fileref]} in this project file, #{canon_ent[:fileref]} in the canonical"
                [ beginning, files, middle, refs, ending ].each do |arr|
                    arr.each { |l| l.gsub!(this_ent[:fileref], canon_ent[:fileref]) }
                end
            end

        end
    end
end



######### Sort FILES by the filename, putting swift etc in front

# key is filename
# value is array of text lines for that filename in the FILES text
# (libraries like libz.dylib seem to occur multiple times, probably
# once each for different targets).

files_by_filename = Hash.new { |k, v| k[v] = Array.new }

files.each do |l|
    # 2669421A1A6DC2AC0063BE93 /* MICmdCmdTarget.cpp in Sources */ = {isa = PBXBuildFile; fileRef = 266941941A6DC2AC0063BE93 /* MICmdCmdTarget.cpp */; };

    if l =~ /^\s+([A-F0-9]{24})\s+\/\*\s+(.*?)\sin.*?\*\/.*?fileRef = ([A-F0-9]{24})\s.*$/
        uuid = $1
        filename = $2
        fileref = $3
        files_by_filename[filename].push(l)
    end

end

# clear the FILES array

files = Array.new

# add the lines in sorted order.  First swift/etc, then everything else.

segregated_filenames.each do |keyword|
    filenames = files_by_filename.keys
    filenames.select {|l| l.include?(keyword) }.sort.each do |fn|
        # re-add all the lines for the filename FN to our FILES array that we'll
        # be outputting.
        files_by_filename[fn].sort.each do |l|
            files.push(l)
        end
        files_by_filename.delete(fn)
    end
end

# All segregated filenames have been added to the FILES output array.
# Now add all the other lines, sorted by filename.

files_by_filename.keys.sort.each do |fn|
    files_by_filename[fn].sort.each do |l|
        files.push(l)
    end
end

######### Sort REFS by the filename, putting swift etc in front

refs_by_filename = Hash.new { |k, v| k[v] = Array.new }
refs.each do |l|
    # 2611FF12142D83060017FEA3 /* SBValue.i */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.c.c.preprocessed; path = SBValue.i; sourceTree = "<group>"; };

    if l =~ /^\s+([A-F0-9]{24})\s+\/\*\s+(.*?)\s\*\/.*$/
        uuid = $1
        filename = $2
        refs_by_filename[filename].push(l)
    end
end

# clear the refs array

refs = Array.new

# add the lines in sorted order.  First swift/etc, then everything else.


segregated_filenames.each do |keyword|
    filenames = refs_by_filename.keys
    filenames.select {|l| l.include?(keyword) }.sort.each do |fn|
        # re-add all the lines for the filename FN to our refs array that we'll
        # be outputting.
        refs_by_filename[fn].sort.each do |l|
            refs.push(l)
        end
        refs_by_filename.delete(fn)
    end
end

# All segregated filenames have been added to the refs output array.
# Now add all the other lines, sorted by filename.

refs_by_filename.keys.sort.each do |fn|
    refs_by_filename[fn].sort.each do |l|
        refs.push(l)
    end
end



####### output the sorted pbxproj

File.open(xcodeproj_filename, 'w') do |outfile|
    [ beginning, files, middle, refs, ending ].each do |arr|
      arr.each {|l| outfile.puts l}
    end
end
