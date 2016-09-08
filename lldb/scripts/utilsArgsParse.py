""" Utility module handle program args and give help

    --------------------------------------------------------------------------
    File:     utilsArgsParse.py

    Overview:  Python module to parse and validate program parameters
               against those required by the program whether mandatory
               or optional.
               Also give help information on arguments required by the
               program.

    Gotchas:   None.

    Copyright: None.
    --------------------------------------------------------------------------

"""

# Python modules:
import getopt  # Parse command line arguments

# Third party modules:

# In-house modules:

# Instantiations:

# User facing text:
strMsgErrorInvalidParameters = "Invalid parameters entered, -h for help. \nYou entered:\n"
strMsgErrorInvalidNoParams = "No parameters entered, -h for help\n"
strMsgErrorNumberParameters = "Number of parameters entered incorrect, %d parameters required. You entered:\n"
strMsgArgFileNotImplemented = "Sorry the --argFile is not implemented"

#++---------------------------------------------------------------------------
# Details: Validate the arguments passed in against the mandatory and
#          optional arguments specified. The argument format for the parameters
#          is required to work with the module getopt function getopt().
#          Parameter vDictArgReq specifies which parameters are mandatory and
#          which are optional. The format is for example:
#            dictArgReq = {"-h": "o", # o = optional, m = mandatory
#                          "-m": "m",
#                          "--targetDir": "m",
#                          "--cfgBldDir": "o" }
# Args:    vArgv - (R) List of arguments and values.
#          vstrListArgs - (R) List of small arguments.
#          vListLongArgs - (R) List of long arguments.
#          vDictArgReq - (R) Map of arguments required.
#          vstrHelpInfo - (R) Formatted help text.
# Returns: Int - 0 success.
#                1 success display information, do nothing else.
#                -1 error invalid parameters.
#                -2 error incorrect number of mandatory parameters.
#          Dict - Map of arguments names to argument values
#          Str - Error message.
# Throws:  None.
#--


def parse(vArgv, vstrListArgs, vListLongArgs, vDictArgReq, vstrHelpInfo):
    dictArgs = {}
    dictDummy = {}
    strDummy = ""

    # Validate parameters above and error on not recognised
    try:
        dictOptsNeeded, dictArgsLeftOver = getopt.getopt(vArgv,
                                                         vstrListArgs,
                                                         vListLongArgs)
    except getopt.GetoptError:
        strMsg = strMsgErrorInvalidParameters
        strMsg += str(vArgv)
        return (-1, dictDummy, strMsg)

    if len(dictOptsNeeded) == 0:
        strMsg = strMsgErrorInvalidNoParams
        return (-1, dictDummy, strMsg)

    # Look for help -h before anything else
    for opt, arg in dictOptsNeeded:
        if opt == '-h':
            return (1, dictDummy, vstrHelpInfo)

    # Look for the --argFile if found ignore other command line arguments
    for opt, arg in dictOptsNeeded:
        if opt == '--argsFile':
            return (1, dictDummy, strMsgArgFileNotImplemented)

    # Count the number of mandatory args required (if any one found)
    countMandatory = 0
    for opt, man in list(vDictArgReq.items()):
        if man == "m":
            countMandatory = countMandatory + 1

    # Extract short args
    listArgs = []
    for arg in vstrListArgs:
        if (arg == '-h') or (arg == ':'):
            continue
        listArgs.append(arg)

    # Append to arg dictionary the option and its value
    bFoundNoInputValue = False
    countMandatoryOpts = 0
    for opt, val in dictOptsNeeded:
        match = 0
        for arg in listArgs:
            argg = "-" + arg
            if opt == argg:
                if "m" == vDictArgReq[opt]:
                    countMandatoryOpts = countMandatoryOpts + 1
                dictArgs[opt] = val
                match = 1
                break
        if match == 0:
            for arg in vListLongArgs:
                argg = "--" + arg[:arg.__len__() - 1]
                if opt == argg:
                    if "m" == vDictArgReq[opt]:
                        countMandatoryOpts = countMandatoryOpts + 1
                    dictArgs[opt] = val
                    if val.__len__() == 0:
                        bFoundNoInputValue = True
                    break

    # Do any of the long arguments not have a value attached
    if bFoundNoInputValue:
        strMsg = strMsgErrorInvalidParameters
        strMsg += str(vArgv)
        return (-1, dictDummy, strMsg)

    # Debug only
    # print countMandatoryOpts
    # print countMandatory

    # Do we have the exact number of mandatory arguments
    if (countMandatoryOpts > 0) and (countMandatory != countMandatoryOpts):
        strMsg = strMsgErrorNumberParameters % countMandatory
        strMsg += str(vArgv)
        return (-2, dictDummy, strMsg)

    return (0, dictArgs, strDummy)
