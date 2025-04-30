#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test gc00  ########


gc00: run
	

build:  $(SRC)/gc00.f
	-$(RM) gc00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/gc00.f -o gc00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) gc00.$(OBJX) check.$(OBJX) $(LIBS) -o gc00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test gc00
	gc00.$(EXESUFFIX)

verify: ;

gc00.run: run

