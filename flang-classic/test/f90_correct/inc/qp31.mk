#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp31  ########


qp31: run


build:  $(SRC)/qp31.f08
	-$(RM) qp31.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp31.f08 -o qp31.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp31.$(OBJX) check.$(OBJX) $(LIBS) -o qp31.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp31
	qp31.$(EXESUFFIX)

verify: ;
