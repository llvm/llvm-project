#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# compile a .c test

build:
	@$(RM) -f $(TEST).$(EXE) core.* *.exe
	@echo ------------------------------------ building compile-only test $@
	$(CC) -c $(CFLAGS) $(SRC)/sources/$(TEST).c -o $(TEST).$(OBJX)
	@echo PASS

run: ;

verify: ;

