#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test nb01  ########


nb01: run
	

build:  $(SRC)/nb01.f
	-$(RM) nb01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/nb01.f -o nb01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) nb01.$(OBJX) check.$(OBJX) $(LIBS) -o nb01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test nb01
	nb01.$(EXESUFFIX)

verify: ;

nb01.run: run

