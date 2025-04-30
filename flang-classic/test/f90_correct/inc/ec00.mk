#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ec00  ########


ec00: run
	

build:  $(SRC)/ec00.f
	-$(RM) ec00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ec00.f -o ec00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ec00.$(OBJX) check.$(OBJX) $(LIBS) -o ec00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ec00
	ec00.$(EXESUFFIX)

verify: ;

ec00.run: run

