# po2test.awk - Convert Uniforum style .po file to C code for testing.
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

# Output current message (in msg) as argument of the INPUT or OUTPUT macro,
# depending on msgtype
function output_message() {
    # Ignore messages containing <PRI.*> markers which would have to be
    # replaced by the correct format depending on the word size
    if (msg && msg !~ /<PRI.*>/)
	printf ("%s(%s)\n", msgtype == "msgid" ? "INPUT" : "OUTPUT", msg)
    msg = 0
}

$1 ~ /msg(id|str)/ {
    # Output collected message
    output_message()
    # Collect next message
    msgtype = $1
    sub(/^msg(id|str)[ \t]*/, "", $0)
    msg = $0
    next
}

/^".*"/ {
    # Append to current message
    msg = msg "\n" $0
}

END {
    # Output last collected message
    output_message()
}
