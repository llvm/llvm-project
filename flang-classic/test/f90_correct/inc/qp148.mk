#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test fold_const  ########


qp148: run


build:  $(SRC)/qp148.f08
	-$(RM) qp148.$(EXESUFFIX) qp148.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp148.f08 -o qp148.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp148.$(OBJX) $(LIBS) -o qp148.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp148
	qp148.$(EXESUFFIX)

verify: ;

qp148.run: run

