#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka05  ########


ka05: run
	

build:  $(SRC)/ka05.f
	-$(RM) ka05.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka05.f -o ka05.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka05.$(OBJX) check.$(OBJX) $(LIBS) -o ka05.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka05
	ka05.$(EXESUFFIX)

verify: ;

ka05.run: run

