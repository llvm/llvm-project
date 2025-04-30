#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test write ########


qp42: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp42.f08 fcheck.$(OBJX)
	-$(RM) qp42.$(EXESUFFIX) qp42.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp42.f08 -o qp42.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp42.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp42.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp42
	qp42.$(EXESUFFIX)

verify: ;

qp42.run: run

