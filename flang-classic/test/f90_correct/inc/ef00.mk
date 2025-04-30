#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ef00  ########


ef00: run
	

build:  $(SRC)/ef00.f
	-$(RM) ef00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ef00.f -o ef00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ef00.$(OBJX) check.$(OBJX) $(LIBS) -o ef00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ef00
	ef00.$(EXESUFFIX)

verify: ;

ef00.run: run

