# xopen-msg.awk - Convert Uniforum style .po file to X/Open style .msg file
# Copyright (C) 2012-2021 Free Software Foundation, Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <https://www.gnu.org/licenses/>.
#
#
# The first directive in the .msg should be the definition of the
# message set number.  We use always set number 1.
#
BEGIN {
    print "$set 1 # Automatically created by xopen-msg.awk"
    num = 0
}

#
# The .msg file contains, other then the .po file, only the translations
# but each given a unique ID.  Starting from 1 and incrementing by 1 for
# each message we assign them to the messages.
# It is important that the .po file used to generate the ../intl/msg.h file
# (with po2test.awk) is the same as the one used here.  (At least the order
# of declarations must not be changed.)
#
function output_message() {
    # Ignore messages containing <PRI.*> which would have to be replaced
    # by the correct format depending on the word size
    if (msg && msg !~ /<PRI.*>/) {
	if (msgtype == "msgid") {
	    # We copy the original message as a comment into the .msg file.
	    gsub(/\n/, "\n$ ", msg)
	    printf "$ Original Message: %s\n", msg
	} else {
	    gsub(/\n/, "\\\n", msg)
	    printf "%d %s\n", ++num, msg
	}
    }
    msg = 0
}

$1 ~ "msg(id|str)" {
    # Output collected message
    output_message()
    # Collect next message
    msgtype = $1
    sub(/^msg(id|str)[ \t]*"/, "", $0)
    sub(/"$/, "", $0)
    msg = $0
    next
}

/^"POT-Creation-Date: [0-9-]+ [0-9:+-]+\\n"/ {
    # Ignore POT-Creation-Date to match what is done in intl/Makefile.
    next
}

/^".*"/ {
    # Append to current message
    sub(/^"/, "", $0)
    sub(/"$/, "", $0)
    msg = msg "\n" $0
    next
}

END {
    # Output last collected message
    output_message()
}
