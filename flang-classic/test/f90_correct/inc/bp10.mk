#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bp10  ########


bp10: run
	

build:  $(SRC)/bp10.f
	-$(RM) bp10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bp10.f -o bp10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bp10.$(OBJX) check.$(OBJX) $(LIBS) -o bp10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bp10
	bp10.$(EXESUFFIX)

verify: ;

bp10.run: run

