#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka03  ########


ka03: run
	

build:  $(SRC)/ka03.f
	-$(RM) ka03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka03.f -o ka03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka03.$(OBJX) check.$(OBJX) $(LIBS) -o ka03.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka03
	ka03.$(EXESUFFIX)

verify: ;

ka03.run: run

