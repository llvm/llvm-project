#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# compile, link, run a .c test

build:
	@$(RM) -f $(TEST).$(EXE) core.* *.exe
	@echo ------------------------------------ building test $@
	$(CC) -c $(CFLAGS) $(SRC)/sources/$(TEST).c -o $(TEST).$(OBJX)
	$(CC) $(TEST).$(OBJX) $(LDFLAGS) -o $(TEST).$(EXE)

run:
	@echo ------------------------------------ executing test $(TEST)
	$(RUN4) $(TEST).$(EXE)
	@echo PASS

verify: ;

