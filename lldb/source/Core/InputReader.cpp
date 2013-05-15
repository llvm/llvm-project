//===-- InputReader.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include <string>

#include "lldb/Core/InputReader.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandInterpreter.h"

using namespace lldb;
using namespace lldb_private;

InputReader::InputReader (Debugger &debugger) :
    m_debugger (debugger),
    m_callback (NULL),
    m_callback_baton (NULL),
    m_end_token (),
    m_granularity (eInputReaderGranularityInvalid),
    m_done (true),
    m_echo (true),
    m_active (false), 
    m_reader_done (false),
    m_user_input(),
    m_save_user_input(false)
{
}

InputReader::~InputReader ()
{
}

Error
InputReader::Initialize 
(
    Callback callback, 
    void *baton,
    lldb::InputReaderGranularity granularity,
    const char *end_token,
    const char *prompt,
    bool echo
)
{
    Error err;
    m_callback = callback;
    m_callback_baton = baton,
    m_granularity = granularity;
    if (end_token != NULL)
        m_end_token = end_token;
    if (prompt != NULL)
        m_prompt = prompt;
    m_done = true;
    m_echo = echo;

    if (m_granularity == eInputReaderGranularityInvalid)
    {
        err.SetErrorString ("Invalid read token size:  Reader must be initialized with a token size other than 'eInputReaderGranularityInvalid'.");
    }
    else
    if (end_token != NULL && granularity != eInputReaderGranularityInvalid)
    {
        if (granularity == eInputReaderGranularityByte)
        {
            // Check to see if end_token is longer than one byte.
            
            if (strlen (end_token) > 1)
            {
                err.SetErrorString ("Invalid end token:  End token cannot be larger than specified token size (byte).");
            }
        }
        else if (granularity == eInputReaderGranularityWord)
        {
            // Check to see if m_end_token contains any white space (i.e. is multiple words).
            
            const char *white_space = " \t\n";
            size_t pos = m_end_token.find_first_of (white_space);
            if (pos != std::string::npos)
            {
                err.SetErrorString ("Invalid end token:  End token cannot be larger than specified token size (word).");
            }
        }
        else
        {
            // Check to see if m_end_token contains any newlines; cannot handle multi-line end tokens.
            
            size_t pos = m_end_token.find_first_of ('\n');
            if (pos != std::string::npos)
            {
                err.SetErrorString ("Invalid end token:  End token cannot contain a newline.");
            }
        }
    }
    
    m_done = err.Fail();

    return err;
}

size_t
InputReader::HandleRawBytes (const char *bytes, size_t bytes_len)
{
    const char *end_token = NULL;
    
    if (m_end_token.empty() == false)
    {
        end_token = ::strstr (bytes, m_end_token.c_str());
        if (end_token >= bytes + bytes_len)
            end_token = NULL;
    }

    const char *p = bytes;
    const char *end = bytes + bytes_len;

    switch (m_granularity)
    {
    case eInputReaderGranularityInvalid:
        break;

    case eInputReaderGranularityByte:
        while (p < end)
        {
            if (end_token == p)
            {
                p += m_end_token.size();
                SetIsDone(true);
                break;
            }

            if (m_callback (m_callback_baton, *this, eInputReaderGotToken, p, 1) == 0)
                break;
            ++p;
            if (IsDone())
                break;
        }
        // Return how many bytes were handled.
        return p - bytes;
        break;


    case eInputReaderGranularityWord:
        {
            char quote = '\0';
            const char *word_start = NULL;
            bool send_word = false;
            for (; p < end; ++p, send_word = false)
            {
                if (end_token && end_token == p)
                {
                    m_end_token.size();
                    SetIsDone(true);
                    break;
                }

                const char ch = *p;
                if (isspace(ch) && (!quote || (quote == ch && p[-1] != '\\')))
                {
                    // We have a space character or the terminating quote
                    send_word = word_start != NULL;
                    quote = '\0';
                }
                else if (quote)
                {
                    // We are in the middle of a quoted character
                    continue;
                }
                else if (ch == '"' || ch == '\'' || ch == '`')
                    quote = ch;
                else if (word_start == NULL)
                {
                    // We have the first character in a word
                    word_start = p;
                }
                
                if (send_word)
                {
                    const size_t word_len = p - word_start;
                    size_t bytes_handled = m_callback (m_callback_baton, 
                                                       *this, 
                                                       eInputReaderGotToken, 
                                                       word_start,
                                                       word_len);

                    if (bytes_handled != word_len)
                        return word_start - bytes + bytes_handled;
                    
                    if (IsDone())
                        return p - bytes;
                }
            }
        }
        break;


    case eInputReaderGranularityLine:
        {
            const char *line_start = bytes;
            const char *end_line = NULL;
            while (p < end)
            {
                const char ch = *p;
                if (ch == '\n' || ch == '\r')
                {
                    size_t line_length = p - line_start;
                    // Now skip the newline character
                    ++p; 
                    // Skip a complete DOS newline if we run into one
                    if (ch == 0xd && p < end && *p == 0xa)
                        ++p;

                    if (line_start <= end_token && end_token < line_start + line_length)
                    {
                        SetIsDone(true);
                        m_callback (m_callback_baton, 
                                    *this, 
                                    eInputReaderGotToken, 
                                    line_start, 
                                    end_token - line_start);
                        
                        return p - bytes;
                    }

                    size_t bytes_handled = m_callback (m_callback_baton, 
                                                       *this, 
                                                       eInputReaderGotToken, 
                                                       line_start, 
                                                       line_length);

                    end_line = p;

                    if (bytes_handled != line_length)
                    {
                        // The input reader wasn't able to handle all the data
                        return line_start - bytes + bytes_handled;
                    }


                    if (IsDone())
                        return p - bytes;

                    line_start = p;
                }
                else
                {
                    ++p;
                }                
            }
            
            if (end_line)
                return end_line - bytes;
        }
        break;

    
    case eInputReaderGranularityAll:
        {
            // Nothing should be handle unless we see our end token
            if (end_token)
            {
                size_t length = end_token - bytes;
                size_t bytes_handled = m_callback (m_callback_baton, 
                                                   *this, 
                                                   eInputReaderGotToken, 
                                                   bytes, 
                                                   length);
                m_done = true;

                p += bytes_handled + m_end_token.size();

                // Consume any white space, such as newlines, beyond the end token

                while (p < end && isspace(*p))
                    ++p;

                if (bytes_handled != length)
                    return bytes_handled;
                else
                {
                    return p - bytes;
                    //return bytes_handled + m_end_token.size();
                }
            }
            return 0;
        }
        break;
    }
    return 0;
}

const char *
InputReader::GetPrompt () const
{
    if (!m_prompt.empty())
        return m_prompt.c_str();
    else
        return NULL;
}

void
InputReader::RefreshPrompt ()
{
	if (m_debugger.GetCommandInterpreter().GetBatchCommandMode())
        return;
    
    if (!m_prompt.empty())
    {
        File &out_file = m_debugger.GetOutputFile();
        if (out_file.IsValid())
        {
            out_file.Printf ("%s", m_prompt.c_str());
            out_file.Flush();
        }
    }
}

void
InputReader::Notify (InputReaderAction notification)
{
    switch (notification)
    {
    case eInputReaderActivate:
    case eInputReaderReactivate:
        m_active = true;
        m_reader_done.SetValue(false, eBroadcastAlways);
        break;

    case eInputReaderDeactivate:
    case eInputReaderDone:
        m_active = false;
        break;
    
    case eInputReaderAsynchronousOutputWritten:
        break;
        
    case eInputReaderInterrupt:
    case eInputReaderEndOfFile:
        break;
    
    case eInputReaderGotToken:
        return; // We don't notify the tokens here, it is done in HandleRawBytes
    }
    if (m_callback)
        m_callback (m_callback_baton, *this, notification, NULL, 0);
    if (notification == eInputReaderDone)
        m_reader_done.SetValue(true, eBroadcastAlways);
}

void 
InputReader::WaitOnReaderIsDone ()
{
    m_reader_done.WaitForValueEqualTo (true);
}

const char *
InputReader::GranularityAsCString (lldb::InputReaderGranularity granularity)
{
    switch (granularity)
    {
    case eInputReaderGranularityInvalid:  return "invalid";
    case eInputReaderGranularityByte:     return "byte";
    case eInputReaderGranularityWord:     return "word";
    case eInputReaderGranularityLine:     return "line";
    case eInputReaderGranularityAll:      return "all";
    }

    static char unknown_state_string[64];
    snprintf(unknown_state_string, sizeof (unknown_state_string), "InputReaderGranularity = %i", granularity);
    return unknown_state_string;
}

bool
InputReader::HandlerData::GetBatchMode()
{
    return reader.GetDebugger().GetCommandInterpreter().GetBatchCommandMode();
}

lldb::StreamSP
InputReader::HandlerData::GetOutStream()
{
    return reader.GetDebugger().GetAsyncOutputStream();
}
