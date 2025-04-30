# Makefile to (re-)generate db versions of system database files.
# Copyright (C) 1996-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
# Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.
#

# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

DATABASES = $(wildcard /etc/passwd /etc/group /etc/ethers /etc/protocols \
		       /etc/rpc /etc/services /etc/shadow /etc/gshadow \
		       /etc/netgroup)

VAR_DB = /var/db

AWK = awk
MAKEDB = makedb --quiet

all: $(patsubst %,$(VAR_DB)/%.db,$(notdir $(DATABASES)))


$(VAR_DB)/passwd.db: /etc/passwd
	@printf %s "$(patsubst %.db,%,$(@F))... "
	@$(AWK) 'BEGIN { FS=":"; OFS=":" } \
		 /^[ \t]*$$/ { next } \
		 /^[ \t]*#/ { next } \
		 /^[^#]/ { printf ".%s ", $$1; print; \
			   printf "=%s ", $$3; print }' $^ | \
	$(MAKEDB) -o $@ -
	@echo "done."

$(VAR_DB)/group.db: /etc/group
	@printf %s "$(patsubst %.db,%,$(@F))... "
	@$(AWK) 'BEGIN { FS=":"; OFS=":" } \
		 /^[ \t]*$$/ { next } \
		 /^[ \t]*#/ { next } \
		 /^[^#]/ { printf ".%s ", $$1; print; \
			   printf "=%s ", $$3; print; \
			   if ($$4 != "") { \
			     split($$4, grmems, ","); \
			     for (memidx in grmems) { \
			       mem=grmems[memidx]; \
			       if (members[mem] == "") \
				 members[mem]=$$3; \
			       else \
				 members[mem]=members[mem] "," $$3; \
			     } \
			     delete grmems; } } \
		 END { for (mem in members) \
			 printf ":%s %s %s\n", mem, mem, members[mem]; }' $^ | \
	$(MAKEDB) -o $@ -
	@echo "done."

$(VAR_DB)/ethers.db: /etc/ethers
	@printf %s "$(patsubst %.db,%,$(@F))... "
	@$(AWK) '/^[ \t]*$$/ { next } \
		 /^[ \t]*#/ { next } \
		 /^[^#]/ { printf ".%s ", $$1; print; \
			   printf "=%s ", $$2; print }' $^ | \
	$(MAKEDB) -o $@ -
	@echo "done."

$(VAR_DB)/protocols.db: /etc/protocols
	@printf %s "$(patsubst %.db,%,$(@F))... "
	@$(AWK) '/^[ \t]*$$/ { next } \
		 /^[ \t]*#/ { next } \
		 /^[^#]/ { printf ".%s ", $$1; print; \
			   printf "=%s ", $$2; print; \
			   for (i = 3; i <= NF && !($$i ~ /^#/); ++i) \
			     { printf ".%s ", $$i; print } }' $^ | \
	$(MAKEDB) -o $@ -
	@echo "done."

$(VAR_DB)/rpc.db: /etc/rpc
	@printf %s "$(patsubst %.db,%,$(@F))... "
	@$(AWK) '/^[ \t]*$$/ { next } \
		 /^[ \t]*#/ { next } \
		 /^[^#]/ { printf ".%s ", $$1; print; \
			   printf "=%s ", $$2; print; \
			   for (i = 3; i <= NF && !($$i ~ /^#/); ++i) \
			     { printf ".%s ", $$i; print } }' $^ | \
	$(MAKEDB) -o $@ -
	@echo "done."

$(VAR_DB)/services.db: /etc/services
	@printf %s "$(patsubst %.db,%,$(@F))... "
	@$(AWK) 'BEGIN { FS="[ \t/]+" } \
		 /^[ \t]*$$/ { next } \
		 /^[ \t]*#/ { next } \
		 /^[^#]/ { sub(/[ \t]*#.*$$/, "");\
			   printf ":%s/%s ", $$1, $$3; print; \
			   printf ":%s/ ", $$1; print; \
			   printf "=%s/%s ", $$2, $$3; print; \
			   printf "=%s/ ", $$2; print; \
			   for (i = 4; i <= NF && !($$i ~ /^#/); ++i) \
			     { printf ":%s/%s ", $$i, $$3; print; \
			       printf ":%s/ ", $$i; print } }' $^ | \
	$(MAKEDB) -o $@ -
	@echo "done."

$(VAR_DB)/shadow.db: /etc/shadow
	@printf %s "$(patsubst %.db,%,$(@F))... "
	@$(AWK) 'BEGIN { FS=":"; OFS=":" } \
		 /^[ \t]*$$/ { next } \
		 /^[ \t]*#/ { next } \
		 /^[^#]/ { printf ".%s ", $$1; print }' $^ | \
	(umask 077 && $(MAKEDB) -o $@ -)
	@echo "done."
	@if chgrp shadow $@ 2>/dev/null; then \
	  chmod g+r $@; \
	else \
	  chown 0 $@; chgrp 0 $@; chmod 600 $@; \
	  echo; \
	  echo "Warning: The shadow password database $@"; \
	  echo "has been set to be readable only by root.  You may want"; \
	  echo "to make it readable by the \`shadow' group depending"; \
	  echo "on your configuration."; \
	  echo; \
	fi

$(VAR_DB)/gshadow.db: /etc/gshadow
	@printf %s "$(patsubst %.db,%,$(@F))... "
	@$(AWK) 'BEGIN { FS=":"; OFS=":" } \
		 /^[ \t]*$$/ { next } \
		 /^[ \t]*#/ { next } \
		 /^[^#]/ { printf ".%s ", $$1; print }' $^ | \
	(umask 077 && $(MAKEDB) -o $@ -)
	@echo "done."
	@if chgrp shadow $@ 2>/dev/null; then \
	  chmod g+r $@; \
	else \
	  chown 0 $@; chgrp 0 $@; chmod 600 $@; \
	  echo; \
	  echo "Warning: The shadow group database $@"; \
	  echo "has been set to be readable only by root.  You may want"; \
	  echo "to make it readable by the \`shadow' group depending"; \
	  echo "on your configuration."; \
	  echo; \
	fi

$(VAR_DB)/netgroup.db: /etc/netgroup
	@printf %s "$(patsubst %.db,%,$(@F))... "
	@$(AWK) 'BEGIN { ini=1 } \
		 /^[ \t]*$$/ { next } \
		 /^[ \t]*#/ { next } \
		 /^[^#]/ { if (sub(/[ \t]*\\$$/, " ") == 0) end="\n"; \
			   else end=""; \
			   gsub(/[ \t]+/, " "); \
			   sub(/^[ \t]*/, ""); \
			   if (ini == 0) printf "%s%s", $$0, end; \
			   else printf ".%s %s%s", $$1, $$0, end; \
			   ini=end == "" ? 0 : 1; } \
			   END { if (ini==0) printf "\n" }' $^ | \
	$(MAKEDB) -o $@ -
	@echo "done."
